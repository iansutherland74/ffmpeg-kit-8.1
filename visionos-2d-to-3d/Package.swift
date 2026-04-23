// swift-tools-version: 5.9
// FFmpegKit 8.1 — visionOS 2D-to-3D Swift Package
// Requires: DepthAnythingV2SmallF16.mlmodelc bundled in the host app target.

import PackageDescription

let package = Package(
    name: "VisionOS2Dto3D",
    platforms: [
        .custom("xros", versionString: "2.0"),
    ],
    products: [
        .library(
            name: "VisionOS2Dto3D",
            targets: ["VisionOS2Dto3D"]
        ),
    ],
    targets: [
        .target(
            name: "VisionOS2Dto3D",
            path: "Sources/VisionOS2Dto3D",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]
        ),
    ]
)
