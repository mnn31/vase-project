// Build:  swiftc -O -o photogrammetry photogrammetry.swift -framework RealityKit
// Run:    ./photogrammetry <input_dir> <output.usdz>
// Uses Apple's PhotogrammetrySession (macOS 12+, requires Apple Silicon).
import Foundation
import RealityKit

guard CommandLine.arguments.count == 3 else {
    print("usage: photogrammetry <input_dir> <output.usdz>")
    exit(2)
}
let inDir = URL(fileURLWithPath: CommandLine.arguments[1])
let outURL = URL(fileURLWithPath: CommandLine.arguments[2])

var config = PhotogrammetrySession.Configuration()
config.featureSensitivity = .normal
config.sampleOrdering = .unordered

let session: PhotogrammetrySession
do {
    session = try PhotogrammetrySession(input: inDir, configuration: config)
} catch {
    print("session init failed: \(error)")
    exit(1)
}

let request = PhotogrammetrySession.Request.modelFile(
    url: outURL,
    detail: .reduced)  // .preview / .reduced / .medium / .full / .raw

let outputHandler = Task {
    do {
        for try await output in session.outputs {
            switch output {
            case .processingComplete:
                print("done")
                return
            case .requestError(_, let err):
                print("error: \(err)")
                exit(1)
            case .requestProgress(_, let frac):
                print(String(format: "progress %.0f%%", frac * 100))
            case .inputComplete:
                print("input ingest complete")
            case .processingCancelled:
                print("cancelled"); exit(1)
            default:
                break
            }
        }
    } catch {
        print("stream error: \(error)")
        exit(1)
    }
}

try session.process(requests: [request])
_ = await outputHandler.value
