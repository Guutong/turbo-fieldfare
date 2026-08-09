import Foundation
import TurboFieldfareRepackCore

public enum AppModelLocation {
    /// Directory name each catalog entry installs into, keyed by
    /// `AppModelCatalogEntry.id`. `gemma4` keeps the pre-catalog default
    /// (`scratch/gemma4.gturbo` / `<AppSupport>/TurboFieldfare/gemma4.gturbo`)
    /// so existing installs and tests are unaffected.
    public static func directoryName(forCatalogID catalogID: String) -> String {
        "\(catalogID).gturbo"
    }

    /// Install directory name for a `ModelSource` (the generalized,
    /// multi-architecture catalog). Gemma keeps `gemma4.gturbo` because
    /// installs predate the catalog and renaming would orphan a ~14 GB
    /// directory on every existing machine; everything else is named from
    /// its source id via `directoryName(forCatalogID:)`.
    static func directoryName(for source: ModelSource) -> String {
        source.id == SupportedModelSource.gemma4_26B_A4B.id
            ? "gemma4.gturbo"
            : directoryName(forCatalogID: source.id)
    }

    public static func defaultURL(forCatalogID catalogID: String = "gemma4") -> URL {
        defaultURL(directoryName: directoryName(forCatalogID: catalogID))
    }

    /// Default install location for a `ModelSource` from the generalized
    /// catalog (Laguna and future architectures alongside Gemma/Qwen3.6).
    static func defaultURL(for source: ModelSource) -> URL {
        defaultURL(directoryName: directoryName(for: source))
    }

    private static func defaultURL(directoryName: String) -> URL {
        let fileManager = FileManager.default
        let applicationSupport = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false)) ?? fileManager.homeDirectoryForCurrentUser
        return resolve(
            explicitURL: nil,
            executableURL: Bundle.main.executableURL,
            currentDirectoryURL: URL(fileURLWithPath: fileManager.currentDirectoryPath,
                                     isDirectory: true),
            applicationSupportURL: applicationSupport,
            fileExists: fileManager.fileExists(atPath:),
            directoryName: directoryName)
    }

    /// Test-facing entry point, keyed by catalog id (matches
    /// `AppModelLocationTests`, which predates the `ModelSource` catalog).
    static func resolve(explicitURL: URL?,
                        executableURL: URL?,
                        currentDirectoryURL: URL,
                        applicationSupportURL: URL,
                        fileExists: (String) -> Bool,
                        catalogID: String = "gemma4") -> URL {
        resolve(explicitURL: explicitURL,
               executableURL: executableURL,
               currentDirectoryURL: currentDirectoryURL,
               applicationSupportURL: applicationSupportURL,
               fileExists: fileExists,
               directoryName: directoryName(forCatalogID: catalogID))
    }

    private static func resolve(explicitURL: URL?,
                        executableURL: URL?,
                        currentDirectoryURL: URL,
                        applicationSupportURL: URL,
                        fileExists: (String) -> Bool,
                        directoryName: String) -> URL {
        if let explicitURL {
            return absoluteURL(explicitURL, relativeTo: currentDirectoryURL)
        }
        if let executableURL,
           let root = packageRoot(startingAt: executableURL.deletingLastPathComponent(),
                                  fileExists: fileExists) {
            return root.appendingPathComponent("scratch/\(directoryName)", isDirectory: true)
                .standardizedFileURL
        }
        if let root = packageRoot(startingAt: currentDirectoryURL, fileExists: fileExists) {
            return root.appendingPathComponent("scratch/\(directoryName)", isDirectory: true)
                .standardizedFileURL
        }
        return applicationSupportURL
            .appendingPathComponent("TurboFieldfare", isDirectory: true)
            .appendingPathComponent(directoryName, isDirectory: true)
            .standardizedFileURL
    }

    private static func absoluteURL(_ url: URL, relativeTo base: URL) -> URL {
        if url.path.hasPrefix("/") {
            return url.standardizedFileURL
        }
        return base.appendingPathComponent(url.path, isDirectory: true).standardizedFileURL
    }

    private static func packageRoot(startingAt start: URL,
                                    fileExists: (String) -> Bool) -> URL? {
        var candidatePath = start.standardizedFileURL.path
        while true {
            let candidate = URL(fileURLWithPath: candidatePath, isDirectory: true)
            let package = candidate.appendingPathComponent("Package.swift").path
            let appSources = candidate.appendingPathComponent(
                "Sources/TurboFieldfareApp/Mac", isDirectory: true).path
            if fileExists(package), fileExists(appSources) {
                return candidate
            }
            let parentPath = (candidatePath as NSString).deletingLastPathComponent
            if parentPath.isEmpty || parentPath == candidatePath { return nil }
            candidatePath = parentPath
        }
    }
}
