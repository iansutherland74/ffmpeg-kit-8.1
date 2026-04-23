// StereoFrameProcessor.swift
// Part of the VisionOS2Dto3D package — ffmpeg-kit-8.1
//
// Converts a 2D CVPixelBuffer into a side-by-side stereo CVPixelBuffer using
// Depth Anything V2 monocular depth estimation.
//
// Three rendering modes are selected automatically based on the disparity setting:
//
//  disparity == 0   → SBS copy (left == right, identity stereo — zero overhead)
//  disparity > 0    → Depth-guided parallax using DepthAnythingV2 at up to 6 Hz
//                     with luminance-fallback parallax between inference frames
//
// The depth inference path throttles to depthInferenceInterval (1/6 s) and caches
// the last good depth map for intermediate frames, keeping audio/video in sync.
// After maxConsecutiveDepthFailures the estimator is bypassed for depthBypassDuration
// seconds to prevent stalls.
//
// Usage:
//   let processor = try StereoFrameProcessor(model: model, maxDisparity: 0.035, temporalSmoothing: 0.7)
//   let frame: StereoProcessedFrame = try processor.process(pixelBuffer: videoFrame)
//   // frame.stereoPixelBuffer is ready to enqueue via StereoSampleBufferBridge

import Foundation
import AVFoundation
import CoreML
import CoreImage
import CoreVideo
import CoreMedia
import Accelerate
import QuartzCore

// MARK: - StereoProcessedFrame

/// A side-by-side stereo CVPixelBuffer produced by StereoFrameProcessor.
public struct StereoProcessedFrame {
    public let stereoPixelBuffer: CVPixelBuffer
}

// MARK: - StereoSampleBufferBridge

/// Thread-safe bridge that wraps an AVSampleBufferVideoRenderer and schedules
/// SBS stereo frames on the host-time clock used by RealityKit's VideoPlayerComponent.
public final class StereoSampleBufferBridge: @unchecked Sendable {
    public let renderer = AVSampleBufferVideoRenderer()

    private var cachedFormatDescription: CMFormatDescription?
    private var cachedDimensions: CMVideoDimensions?
    // Monotonic presentation clock anchored to host (mach absolute) time.
    private var nextScheduledTime: CMTime = .invalid
    private var lastEnqueueHostTime: CMTime = .invalid
    private static let lookahead = CMTime(seconds: 0.028, preferredTimescale: 1_000_000)

    public init() {}

    /// Flush the renderer and reset the scheduling state (call when reattaching).
    public func flushForReattach() {
        renderer.flush()
        cachedFormatDescription = nil
        cachedDimensions = nil
        nextScheduledTime = .invalid
        lastEnqueueHostTime = .invalid
    }

    /// Enqueue a stereo pixel buffer for display at the next scheduled host-time slot.
    /// The `duration` parameter should match the processing cadence (default 1/12 s).
    public func enqueue(pixelBuffer: CVPixelBuffer, at presentationTime: CMTime, duration: CMTime = .invalid) throws {
        let hostNow = CMClockGetTime(CMClockGetHostTimeClock())
        let earliest = CMTimeAdd(hostNow, Self.lookahead)

        lastEnqueueHostTime = hostNow
        let frameDuration = (duration.isValid && duration != .zero) ? duration : CMTime(value: 1, timescale: 12)

        // Cap scheduling drift: no more than 3 frame durations ahead.
        let threeFrames = CMTimeAdd(CMTimeAdd(frameDuration, frameDuration), frameDuration)
        let maxLead = CMTimeAdd(earliest, threeFrames)
        let scheduleTime: CMTime
        if nextScheduledTime.isValid &&
           CMTimeCompare(nextScheduledTime, earliest) > 0 &&
           CMTimeCompare(nextScheduledTime, maxLead) <= 0 {
            scheduleTime = nextScheduledTime
        } else {
            scheduleTime = earliest
        }
        nextScheduledTime = CMTimeAdd(scheduleTime, frameDuration)

        let formatDescription = try getFormatDescription(for: pixelBuffer)
        let timing = CMSampleTimingInfo(
            duration: frameDuration,
            presentationTimeStamp: scheduleTime,
            decodeTimeStamp: .invalid
        )
        let sampleBuffer = try CMSampleBuffer(
            imageBuffer: pixelBuffer,
            formatDescription: formatDescription,
            sampleTiming: timing
        )
        renderer.enqueue(sampleBuffer)
    }

    public func flush() {
        renderer.flush()
        nextScheduledTime = .invalid
        lastEnqueueHostTime = .invalid
    }

    // MARK: Private

    private func getFormatDescription(for pixelBuffer: CVPixelBuffer) throws -> CMFormatDescription {
        let dimensions = CMVideoDimensions(
            width: Int32(CVPixelBufferGetWidth(pixelBuffer)),
            height: Int32(CVPixelBufferGetHeight(pixelBuffer))
        )

        if let cachedFormatDescription,
           let cachedDimensions,
           cachedDimensions.width == dimensions.width,
           cachedDimensions.height == dimensions.height {
            return cachedFormatDescription
        }

        let baseFormat = try CMVideoFormatDescription(imageBuffer: pixelBuffer)
        var extensions = baseFormat.extensions

        if #available(visionOS 26.0, *) {
            extensions[.viewPackingKind] = .viewPackingKind(.sideBySide)
            extensions[.projectionKind] = .projectionKind(.rectilinear)
            extensions[.horizontalFieldOfView] = .number(UInt32(65_000))
        }

        let formatDescription = try CMVideoFormatDescription(
            videoCodecType: baseFormat.mediaSubType,
            width: Int(baseFormat.dimensions.width),
            height: Int(baseFormat.dimensions.height),
            extensions: extensions
        )

        cachedFormatDescription = formatDescription
        cachedDimensions = dimensions
        return formatDescription
    }
}

// MARK: - StereoFrameProcessor

/// Converts a 2D CVPixelBuffer to a side-by-side stereo CVPixelBuffer.
///
/// - Parameter maxDisparity: Controls the perceived depth. 0 = flat SBS copy.
///   Typical values: 0.02–0.05 (maps to ~3–7 px shift on 1080p input).
/// - Parameter temporalSmoothing: EMA alpha for the depth map (0 = off, 0.7 = smooth).
public final class StereoFrameProcessor: @unchecked Sendable {
    private let estimator: DepthAnythingEstimator

    // Explicit sRGB output prevents Vision Pro's wide-gamut display from washing out colours.
    private let ciContext = CIContext(options: [
        .workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB) as Any,
        .outputColorSpace: CGColorSpace(name: CGColorSpace.sRGB) as Any,
    ])

    private let disparityControlQueue = DispatchQueue(label: "com.visionos2dto3d.disparity-control")
    private var runtimeMaxDisparity: CGFloat
    private let temporalSmoothing: Float
    private var depthEMA: MLMultiArray?

    // Pixel buffer pools (reused across frames to avoid per-frame allocation).
    private var stereoPixelBufferPool: CVPixelBufferPool?
    private var stereoPoolWidth: Int = 0
    private var stereoPoolHeight: Int = 0
    private var depthPixelBufferPool: CVPixelBufferPool?
    private var depthPoolWidth: Int = 0
    private var depthPoolHeight: Int = 0

    // Depth inference rate control and failure recovery.
    private var cachedDepthForReuse: MLMultiArray?
    private var lastDepthInferenceHostTime: CFTimeInterval = 0
    private let depthInferenceInterval: CFTimeInterval = 1.0 / 6.0
    private var consecutiveDepthFailures = 0
    private var depthBypassUntilHostTime: CFTimeInterval = 0
    private let maxConsecutiveDepthFailures = 3
    private let depthBypassDuration: CFTimeInterval = 2.0

    public init(model: MLModel, maxDisparity: CGFloat, temporalSmoothing: Float) throws {
        self.estimator = try DepthAnythingEstimator(model: model)
        self.runtimeMaxDisparity = max(0, maxDisparity)
        self.temporalSmoothing = min(max(temporalSmoothing, 0), 0.99)
    }

    /// Thread-safe update of the disparity setting during playback.
    public func updateMaxDisparity(_ value: CGFloat) {
        disparityControlQueue.sync {
            runtimeMaxDisparity = max(0, value)
        }
    }

    /// Convert a 2D CVPixelBuffer to a side-by-side stereo CVPixelBuffer.
    public func process(pixelBuffer: CVPixelBuffer) throws -> StereoProcessedFrame {
        let activeDisparity = disparityControlQueue.sync { runtimeMaxDisparity }

        // disparity == 0: flat SBS copy — no depth work needed.
        if activeDisparity <= 0.001 {
            return try makeStereoSBSWithoutDepth(pixelBuffer: pixelBuffer)
        }

        // Check if inference is in cooldown after repeated failures.
        let now = CACurrentMediaTime()
        if depthBypassUntilHostTime > now {
            return try makeStereoSBSParallax(pixelBuffer: pixelBuffer, disparity: activeDisparity)
        }

        // Depth inference path — throttled to depthInferenceInterval (1/6 s).
        let shouldRunInference = (now - lastDepthInferenceHostTime) >= depthInferenceInterval
        if shouldRunInference {
            do {
                let rawDepth = try estimator.predictDepth(pixelBuffer: pixelBuffer)
                let smoothed = try smoothDepth(rawDepth)
                cachedDepthForReuse = smoothed
                lastDepthInferenceHostTime = now
                consecutiveDepthFailures = 0
            } catch {
                consecutiveDepthFailures += 1
                if consecutiveDepthFailures >= maxConsecutiveDepthFailures {
                    depthBypassUntilHostTime = now + depthBypassDuration
                    consecutiveDepthFailures = 0
                    cachedDepthForReuse = nil
                }
            }
        }

        if let depth = cachedDepthForReuse {
            return try makeStereoSBS(pixelBuffer: pixelBuffer, depth: depth, disparity: activeDisparity)
        }

        // No cached depth yet (first frame or post-bypass warm-up).
        return try makeStereoSBSParallax(pixelBuffer: pixelBuffer, disparity: activeDisparity)
    }

    // MARK: - Rendering modes

    /// Flat SBS copy — left and right are identical (disparity == 0 fast path).
    private func makeStereoSBSWithoutDepth(pixelBuffer: CVPixelBuffer) throws -> StereoProcessedFrame {
        let source = CIImage(cvPixelBuffer: pixelBuffer, options: [.colorSpace: Self.bt709])
        let width = source.extent.width
        let height = source.extent.height

        let canvasRect = CGRect(x: 0, y: 0, width: width * 2, height: height)
        let background = CIImage(color: .black).cropped(to: canvasRect)
        let rightPlaced = source.transformed(by: CGAffineTransform(translationX: width, y: 0))
        let composed = rightPlaced.composited(over: source).composited(over: background)

        return try renderToStereoBuffer(composed, width: Int(width * 2), height: Int(height))
    }

    /// Luminance-mask pseudo-parallax — no ML, used between inference frames and during warm-up.
    private func makeStereoSBSParallax(pixelBuffer: CVPixelBuffer, disparity: CGFloat) throws -> StereoProcessedFrame {
        let source = CIImage(cvPixelBuffer: pixelBuffer, options: [.colorSpace: Self.bt709])
        let width = source.extent.width
        let height = source.extent.height

        let shiftPixels = min(max(disparity * 150.0, 0), 150.0)
        let nearShift = shiftPixels * 0.5
        let farShift = shiftPixels * 0.2

        let pseudoDepthMask = source
            .applyingFilter("CIColorControls", parameters: [
                kCIInputSaturationKey: 0.0,
                kCIInputContrastKey: 1.25,
                kCIInputBrightnessKey: 0.0,
            ])
            .clampedToExtent()
            .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 3.0])
            .cropped(to: source.extent)

        let left = source
            .transformed(by: CGAffineTransform(translationX: nearShift, y: 0))
            .applyingFilter("CIBlendWithMask", parameters: [
                kCIInputBackgroundImageKey: source.transformed(by: CGAffineTransform(translationX: -farShift, y: 0)),
                kCIInputMaskImageKey: pseudoDepthMask,
            ])
            .cropped(to: source.extent)

        let right = source
            .transformed(by: CGAffineTransform(translationX: -nearShift, y: 0))
            .applyingFilter("CIBlendWithMask", parameters: [
                kCIInputBackgroundImageKey: source.transformed(by: CGAffineTransform(translationX: farShift, y: 0)),
                kCIInputMaskImageKey: pseudoDepthMask,
            ])
            .cropped(to: source.extent)

        let canvasRect = CGRect(x: 0, y: 0, width: width * 2, height: height)
        let background = CIImage(color: .black).cropped(to: canvasRect)
        let rightPlaced = right.transformed(by: CGAffineTransform(translationX: width, y: 0))
        let composed = rightPlaced.composited(over: left).composited(over: background)

        return try renderToStereoBuffer(composed, width: Int(width * 2), height: Int(height))
    }

    /// Full depth-guided SBS — uses the DepthAnything depth map as a blend mask.
    private func makeStereoSBS(pixelBuffer: CVPixelBuffer, depth: MLMultiArray, disparity: CGFloat) throws -> StereoProcessedFrame {
        let source = CIImage(cvPixelBuffer: pixelBuffer, options: [.colorSpace: Self.bt709])
        let width = source.extent.width
        let height = source.extent.height

        let depthImage = try makeNormalizedDepthImage(from: depth)
            .transformed(by: CGAffineTransform(
                scaleX: width / depthWidth(depth),
                y: height / depthHeight(depth)
            ))
            .clampedToExtent()
            .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 2.8])
            .cropped(to: source.extent)

        let maxShiftPixels = min(max(disparity * 150.0, 0), 150.0)
        let nearShift = maxShiftPixels * 0.5
        let farShift = maxShiftPixels * 0.2

        let left = source
            .transformed(by: CGAffineTransform(translationX: nearShift, y: 0))
            .applyingFilter("CIBlendWithMask", parameters: [
                kCIInputBackgroundImageKey: source.transformed(by: CGAffineTransform(translationX: -farShift, y: 0)),
                kCIInputMaskImageKey: depthImage,
            ])
            .cropped(to: source.extent)

        let right = source
            .transformed(by: CGAffineTransform(translationX: -nearShift, y: 0))
            .applyingFilter("CIBlendWithMask", parameters: [
                kCIInputBackgroundImageKey: source.transformed(by: CGAffineTransform(translationX: farShift, y: 0)),
                kCIInputMaskImageKey: depthImage,
            ])
            .cropped(to: source.extent)

        let canvasRect = CGRect(x: 0, y: 0, width: width * 2, height: height)
        let background = CIImage(color: .black).cropped(to: canvasRect)
        let rightPlaced = right.transformed(by: CGAffineTransform(translationX: width, y: 0))
        let composed = rightPlaced.composited(over: left).composited(over: background)

        return try renderToStereoBuffer(composed, width: Int(width * 2), height: Int(height))
    }

    // MARK: - Render helpers

    private static let bt709 = CGColorSpace(name: CGColorSpace.itur_709)!

    private func renderToStereoBuffer(_ image: CIImage, width: Int, height: Int) throws -> StereoProcessedFrame {
        let stereoPixelBuffer = try makeReusableStereoPixelBuffer(width: width, height: height)
        let colorAttachments: [CFString: Any] = [
            kCVImageBufferColorPrimariesKey: kCVImageBufferColorPrimaries_ITU_R_709_2,
            kCVImageBufferTransferFunctionKey: kCVImageBufferTransferFunction_ITU_R_709_2,
            kCVImageBufferYCbCrMatrixKey: kCVImageBufferYCbCrMatrix_ITU_R_709_2,
        ]
        CVBufferSetAttachments(stereoPixelBuffer, colorAttachments as CFDictionary, .shouldPropagate)
        ciContext.render(
            image,
            to: stereoPixelBuffer,
            bounds: CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)),
            colorSpace: Self.bt709
        )
        return StereoProcessedFrame(stereoPixelBuffer: stereoPixelBuffer)
    }

    // MARK: - Temporal smoothing (EMA via Accelerate)

    private func smoothDepth(_ depth: MLMultiArray) throws -> MLMultiArray {
        if temporalSmoothing <= 0.001 {
            depthEMA = depth
            return depth
        }
        guard depth.dataType == .float32 else {
            depthEMA = depth
            return depth
        }
        let ema: MLMultiArray
        if let prev = depthEMA, prev.count == depth.count, prev.dataType == .float32 {
            ema = prev
        } else {
            ema = try MLMultiArray(shape: depth.shape, dataType: .float32)
            let src = depth.dataPointer.bindMemory(to: Float.self, capacity: depth.count)
            let dst = ema.dataPointer.bindMemory(to: Float.self, capacity: depth.count)
            dst.update(from: src, count: depth.count)
            depthEMA = ema
            return ema
        }
        var alpha = temporalSmoothing
        var oneMinusAlpha = 1.0 - alpha
        let depthPtr = depth.dataPointer.bindMemory(to: Float.self, capacity: depth.count)
        let emaPtr = ema.dataPointer.bindMemory(to: Float.self, capacity: ema.count)
        let n = vDSP_Length(depth.count)
        vDSP_vsmul(emaPtr, 1, &alpha, emaPtr, 1, n)
        vDSP_vsma(depthPtr, 1, &oneMinusAlpha, emaPtr, 1, emaPtr, 1, n)
        depthEMA = ema
        return ema
    }

    // MARK: - Depth image normalisation

    private func makeNormalizedDepthImage(from depth: MLMultiArray) throws -> CIImage {
        guard depth.dataType == .float32 || depth.dataType == .float16 || depth.dataType == .double else {
            throw DepthEstimatorError.unexpectedOutputType
        }
        let shape = depth.shape.map { $0.intValue }
        guard shape.count >= 2 else { throw DepthEstimatorError.unexpectedOutputShape }

        let depthH = shape[shape.count - 2]
        let depthW = shape[shape.count - 1]
        let count = depthH * depthW
        guard count > 0 else { throw DepthEstimatorError.unexpectedOutputShape }

        var minValue = Float.greatestFiniteMagnitude
        var maxValue = -Float.greatestFiniteMagnitude
        for i in 0..<count {
            let v = depth[i].floatValue
            if v < minValue { minValue = v }
            if v > maxValue { maxValue = v }
        }
        let denom = max(maxValue - minValue, 1e-6)
        let pixelBuffer = try makeReusableDepthPixelBuffer(width: depthW, height: depthH)

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw DepthEstimatorError.imageCreationFailed
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let dst = baseAddress.assumingMemoryBound(to: UInt8.self)
        for y in 0..<depthH {
            let row = dst.advanced(by: y * bytesPerRow)
            let rowOffset = y * depthW
            for x in 0..<depthW {
                let normalized = (depth[rowOffset + x].floatValue - minValue) / denom
                row[x] = UInt8(max(0, min(255, Int(normalized * 255.0))))
            }
        }
        return CIImage(cvPixelBuffer: pixelBuffer)
    }

    private func depthWidth(_ depth: MLMultiArray) -> CGFloat {
        let shape = depth.shape.map { $0.intValue }
        return shape.count >= 2 ? CGFloat(shape[shape.count - 1]) : 1
    }

    private func depthHeight(_ depth: MLMultiArray) -> CGFloat {
        let shape = depth.shape.map { $0.intValue }
        return shape.count >= 2 ? CGFloat(shape[shape.count - 2]) : 1
    }

    // MARK: - Pixel buffer pool management

    private func makeReusableStereoPixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        if stereoPixelBufferPool == nil || stereoPoolWidth != width || stereoPoolHeight != height {
            let pixelBufferAttributes: [CFString: Any] = [
                kCVPixelBufferPixelFormatTypeKey: Int(kCVPixelFormatType_32BGRA),
                kCVPixelBufferWidthKey: width,
                kCVPixelBufferHeightKey: height,
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
                kCVPixelBufferMetalCompatibilityKey: true,
                kCVImageBufferColorPrimariesKey: kCVImageBufferColorPrimaries_ITU_R_709_2,
                kCVImageBufferTransferFunctionKey: kCVImageBufferTransferFunction_ITU_R_709_2,
            ]
            let poolAttributes: [CFString: Any] = [kCVPixelBufferPoolMinimumBufferCountKey: 3]
            var pool: CVPixelBufferPool?
            guard CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttributes as CFDictionary, pixelBufferAttributes as CFDictionary, &pool) == kCVReturnSuccess, let pool else {
                throw NSError(domain: "StereoFrameProcessor", code: -2)
            }
            stereoPixelBufferPool = pool
            stereoPoolWidth = width
            stereoPoolHeight = height
        }
        guard let pool = stereoPixelBufferPool else { throw NSError(domain: "StereoFrameProcessor", code: -3) }
        var buf: CVPixelBuffer?
        let aux: [CFString: Any] = [kCVPixelBufferPoolAllocationThresholdKey: 6]
        let status = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(kCFAllocatorDefault, pool, aux as CFDictionary, &buf)
        if status == kCVReturnWouldExceedAllocationThreshold { throw NSError(domain: "StereoFrameProcessor", code: -5) }
        guard status == kCVReturnSuccess, let buf else { throw NSError(domain: "StereoFrameProcessor", code: -4) }
        return buf
    }

    private func makeReusableDepthPixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        if depthPixelBufferPool == nil || depthPoolWidth != width || depthPoolHeight != height {
            let pixelBufferAttributes: [CFString: Any] = [
                kCVPixelBufferPixelFormatTypeKey: Int(kCVPixelFormatType_OneComponent8),
                kCVPixelBufferWidthKey: width,
                kCVPixelBufferHeightKey: height,
                kCVPixelBufferMetalCompatibilityKey: true,
            ]
            let poolAttributes: [CFString: Any] = [kCVPixelBufferPoolMinimumBufferCountKey: 2]
            var pool: CVPixelBufferPool?
            guard CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttributes as CFDictionary, pixelBufferAttributes as CFDictionary, &pool) == kCVReturnSuccess, let pool else {
                throw NSError(domain: "StereoFrameProcessor", code: -6)
            }
            depthPixelBufferPool = pool
            depthPoolWidth = width
            depthPoolHeight = height
        }
        guard let pool = depthPixelBufferPool else { throw NSError(domain: "StereoFrameProcessor", code: -7) }
        var buf: CVPixelBuffer?
        let aux: [CFString: Any] = [kCVPixelBufferPoolAllocationThresholdKey: 4]
        let status = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(kCFAllocatorDefault, pool, aux as CFDictionary, &buf)
        if status == kCVReturnWouldExceedAllocationThreshold { throw NSError(domain: "StereoFrameProcessor", code: -8) }
        guard status == kCVReturnSuccess, let buf else { throw NSError(domain: "StereoFrameProcessor", code: -9) }
        return buf
    }
}
