import Foundation

public protocol AppModelInstallerClient: Sendable {
    var descriptor: AppModelInstallDescriptor { get }
    /// Replace the wrapped descriptor so a new model can be installed without
    /// recreating the client. Implementations must cancel any in-flight work
    /// before rebinding.
    func rebind(descriptor: AppModelInstallDescriptor)
    func checkInstallRequirement(outputDirectory: URL) throws -> AppModelInstallRequirement
    func installDefaultModel(outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error>
    func discardPartialInstall(outputDirectory: URL) async throws
    func cancel()
}
