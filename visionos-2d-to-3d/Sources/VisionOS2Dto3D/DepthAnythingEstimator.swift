// DepthAnythingEstimator.swift
// Part of the VisionOS2Dto3D package — ffmpeg-kit-8.1
//
// Wraps a Depth Anything V2 Core ML model (DepthAnythingV2SmallF16.mlmodelc) for
// monocular depth estimation on visionOS.  The model must be compiled and bundled
// in the host application target.
//
// Usage:
//   let config = MLModelConfiguration()
//   config.computeUnits = .all
//   let model = try DepthAnythingModelLoader.loadBundledModel(configuration: config)
//   let estimator = try DepthAnythingEstimator(model: model)
//   let depth: MLMultiArray = try estimator.predictDepth(pixelBuffer: frame)

import Foundation
import CoreML
import Vision
import CoreImage
import CoreVideo

public enum DepthAnythingModelLoader {
    /// Loads the compiled DepthAnythingV2SmallF16.mlmodelc from the main bundle.
    public static func loadBundledModel(configuration: MLModelConfiguration) throws -> MLModel {
        guard let modelURL = Bundle.main.url(forResource: "DepthAnythingV2SmallF16", withExtension: "mlmodelc") else {
            throw DepthEstimatorError.modelNotFound
        }
        return try MLModel(contentsOf: modelURL, configuration: configuration)
    }
}

/// Lightweight wrapper around a Depth Anything V2 Core ML model for visionOS.
///
/// - Thread safety: instances are not thread-safe. Create one per processing queue.
public final class DepthAnythingEstimator {
    private let request: VNCoreMLRequest
    private let ciContext = CIContext()
    private let maxInferenceDimension = 640
    private var inferencePixelBufferPool: CVPixelBufferPool?
    private var inferencePoolWidth = 0
    private var inferencePoolHeight = 0
    private var inferencePoolPixelFormat: OSType = 0

    public init(model: MLModel) throws {
        let vnModel = try VNCoreMLModel(for: model)
        request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .scaleFill
    }

    public convenience init(fromGeneratedModel generatedModel: MLModel) throws {
        try self.init(model: generatedModel)
    }

    /// Runs depth inference on a CVPixelBuffer and returns the depth map as an MLMultiArray.
    /// Inputs larger than 640px on the longest edge are downscaled before inference to keep
    /// latency predictable.
    public func predictDepth(pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        let inferencePixelBuffer = try makeInferencePixelBuffer(from: pixelBuffer)
        let handler = VNImageRequestHandler(cvPixelBuffer: inferencePixelBuffer, orientation: .up)
        try handler.perform([request])

        guard let result = request.results?.first else {
            throw DepthEstimatorError.noResult
        }

        if let observation = result as? VNCoreMLFeatureValueObservation,
           let array = observation.featureValue.multiArrayValue {
            return array
        }

        if let observation = result as? VNCoreMLFeatureValueObservation,
           let imageBuffer = observation.featureValue.imageBufferValue {
            return try makeDepthArray(from: imageBuffer)
        }

        if let observation = result as? VNPixelBufferObservation {
            return try makeDepthArray(from: observation.pixelBuffer)
        }

        throw DepthEstimatorError.unsupportedObservation(String(describing: type(of: result)))
    }

    // MARK: - Internal helpers

    /// Scales the input down to at most 640px on the longest edge using a pooled CVPixelBuffer.
    /// Returns the original buffer unchanged if it already fits within the limit.
    private func makeInferencePixelBuffer(from pixelBuffer: CVPixelBuffer) throws -> CVPixelBuffer {
        let sourceWidth = CVPixelBufferGetWidth(pixelBuffer)
        let sourceHeight = CVPixelBufferGetHeight(pixelBuffer)
        let longestEdge = max(sourceWidth, sourceHeight)

        guard longestEdge > maxInferenceDimension else {
            return pixelBuffer
        }

        let scale = Double(maxInferenceDimension) / Double(longestEdge)
        var targetWidth = max(1, Int((Double(sourceWidth) * scale).rounded()))
        var targetHeight = max(1, Int((Double(sourceHeight) * scale).rounded()))

        // Keep dimensions even to avoid odd-size edge cases in downstream kernels.
        targetWidth -= targetWidth % 2
        targetHeight -= targetHeight % 2
        if targetWidth <= 0 { targetWidth = 2 }
        if targetHeight <= 0 { targetHeight = 2 }

        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        if inferencePixelBufferPool == nil ||
            inferencePoolWidth != targetWidth ||
            inferencePoolHeight != targetHeight ||
            inferencePoolPixelFormat != pixelFormat {
            let pixelAttributes: [CFString: Any] = [
                kCVPixelBufferWidthKey: targetWidth,
                kCVPixelBufferHeightKey: targetHeight,
                kCVPixelBufferPixelFormatTypeKey: pixelFormat,
                kCVPixelBufferIOSurfacePropertiesKey: [:],
                kCVPixelBufferMetalCompatibilityKey: true,
            ]
            let poolAttributes: [CFString: Any] = [
                kCVPixelBufferPoolMinimumBufferCountKey: 2,
                kCVPixelBufferPoolAllocationThresholdKey: 4,
            ]
            var pool: CVPixelBufferPool?
            let status = CVPixelBufferPoolCreate(
                kCFAllocatorDefault,
                poolAttributes as CFDictionary,
                pixelAttributes as CFDictionary,
                &pool
            )
            guard status == kCVReturnSuccess, let pool else {
                throw DepthEstimatorError.imageCreationFailed
            }
            inferencePixelBufferPool = pool
            inferencePoolWidth = targetWidth
            inferencePoolHeight = targetHeight
            inferencePoolPixelFormat = pixelFormat
        }

        guard let pool = inferencePixelBufferPool else {
            throw DepthEstimatorError.imageCreationFailed
        }

        var output: CVPixelBuffer?
        let allocStatus = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &output)
        guard allocStatus == kCVReturnSuccess, let output else {
            return pixelBuffer
        }

        let sourceImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scaledImage = sourceImage.transformed(by: CGAffineTransform(
            scaleX: CGFloat(targetWidth) / CGFloat(sourceWidth),
            y: CGFloat(targetHeight) / CGFloat(sourceHeight)
        ))
        ciContext.render(
            scaledImage,
            to: output,
            bounds: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight),
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
        )
        return output
    }

    private func makeDepthArray(from pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let array = try MLMultiArray(
            shape: [NSNumber(value: height), NSNumber(value: width)],
            dataType: .float32
        )

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let planeIndex = CVPixelBufferGetPlaneCount(pixelBuffer) > 0 ? 0 : -1
        let baseAddress: UnsafeMutableRawPointer?
        let bytesPerRow: Int

        if planeIndex >= 0 {
            baseAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, planeIndex)
            bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, planeIndex)
        } else {
            baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
            bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        }

        guard let baseAddress else {
            throw DepthEstimatorError.imageCreationFailed
        }

        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)

        switch pixelFormat {
        case kCVPixelFormatType_OneComponent16Half:
            let rowStride = bytesPerRow / MemoryLayout<UInt16>.stride
            let buffer = baseAddress.assumingMemoryBound(to: UInt16.self)
            for y in 0..<height {
                let row = buffer.advanced(by: y * rowStride)
                for x in 0..<width {
                    array[y * width + x] = NSNumber(value: Float(Float16(bitPattern: row[x])))
                }
            }
        case kCVPixelFormatType_OneComponent16:
            let rowStride = bytesPerRow / MemoryLayout<UInt16>.stride
            let buffer = baseAddress.assumingMemoryBound(to: UInt16.self)
            for y in 0..<height {
                let row = buffer.advanced(by: y * rowStride)
                for x in 0..<width {
                    array[y * width + x] = NSNumber(value: Float(row[x]) / Float(UInt16.max))
                }
            }
        case kCVPixelFormatType_OneComponent8:
            let rowStride = bytesPerRow / MemoryLayout<UInt8>.stride
            let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
            for y in 0..<height {
                let row = buffer.advanced(by: y * rowStride)
                for x in 0..<width {
                    array[y * width + x] = NSNumber(value: Float(row[x]) / Float(UInt8.max))
                }
            }
        default:
            throw DepthEstimatorError.unsupportedPixelFormat(pixelFormat)
        }

        return array
    }
}

// MARK: - Errors

public enum DepthEstimatorError: Error, LocalizedError {
    case noResult
    case unexpectedOutputType
    case unexpectedOutputShape
    case imageCreationFailed
    case modelNotFound
    case unsupportedObservation(String)
    case unsupportedPixelFormat(OSType)

    public var errorDescription: String? {
        switch self {
        case .noResult:                              return "No depth estimation result"
        case .unexpectedOutputType:                  return "Unexpected output data type"
        case .unexpectedOutputShape:                 return "Unexpected output shape"
        case .imageCreationFailed:                   return "Failed to create image"
        case .modelNotFound:                         return "Bundled DepthAnythingV2SmallF16.mlmodelc not found in app bundle"
        case let .unsupportedObservation(t):         return "Unsupported depth observation type: \(t)"
        case let .unsupportedPixelFormat(f):         return "Unsupported depth pixel format: \(f)"
        }
    }
}
