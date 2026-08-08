import Foundation
import TurboFieldfare

public enum AppModelInstallationStatus: Equatable, Sendable {
    case missing
    case partial(String)
    case complete
}

public enum AppModelInstallationProbe {
    public static func status(
        at directory: URL,
        descriptor: AppModelInstallDescriptor = .default
    ) -> AppModelInstallationStatus {
        let directory = directory.standardizedFileURL
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return .missing
        }

        do {
            // Auto-detect model family from manifest arch before validation.
            let manifestData = try Data(contentsOf: manifestURL)
            let expectedArch: ArchConfig
            if let root = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
               let arch = root["arch"] as? [String: Any] {
                expectedArch = ArchConfig.detect(
                    hiddenSize: arch["hiddenSize"] as? Int ?? 0,
                    numLayers: arch["numLayers"] as? Int ?? 0,
                    numExperts: arch["numExperts"] as? Int ?? 0,
                    numKVHeads: arch["numKVHeads"] as? Int ?? 0,
                    numFullKVHeads: arch["numFullKVHeads"] as? Int ?? 0)
            } else {
                expectedArch = .gemma4_26B_A4B
            }
            let manifest = try ManifestReader.load(directoryURL: directory, expecting: expectedArch)
            let expectedSource = "sha256:" + descriptor.sourceIndexSHA256
            guard manifest.sourceSnapshotHash == expectedSource else {
                return .partial("installed checkpoint does not match \(descriptor.displayName)")
            }
            let layout = directory.appendingPathComponent("packed_experts/layout.json")
            guard FileManager.default.fileExists(atPath: layout.path) else {
                return .partial("packed_experts/layout.json is missing")
            }
            let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directory)
            let manifestHash = try Sha256Verifier.hashFile(at: manifestURL, chunkBytes: 65_536)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                receipt,
                directoryURL: directory,
                manifestSha256: manifestHash)
            return .complete
        } catch {
            return .partial("\(error)")
        }
    }
}
