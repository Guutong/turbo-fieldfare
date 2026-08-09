import Foundation

/// One row of HF Hub's public model search response, trimmed to the fields
/// the browse UI needs. Not a full mirror of the Hub API's schema — see
/// `HuggingFaceModelSummary.init(from:)` for the exact fields decoded.
public struct HuggingFaceModelSummary: Equatable, Identifiable, Sendable {
    public var id: String { repoID }
    public let repoID: String
    public let downloads: Int
    public let likes: Int
    public let pipelineTag: String?
    public let tags: [String]

    public init(repoID: String, downloads: Int, likes: Int, pipelineTag: String?, tags: [String]) {
        self.repoID = repoID
        self.downloads = downloads
        self.likes = likes
        self.pipelineTag = pipelineTag
        self.tags = tags
    }
}

/// Errors surfaced by `HuggingFaceModelSearchClient`.
public enum HuggingFaceModelSearchError: Error, Equatable, Sendable {
    case invalidResponse
    case httpStatus(Int)
    case decodingFailed
}

/// Browse-only client for HF Hub's public model search REST endpoint
/// (`GET /api/models`). No authentication, no download — this only lists
/// what's on the Hub so the UI can cross-reference it against
/// `AppModelCatalog`. Actually installing a model still goes through the
/// app's existing `.gturbo` repack/install pipeline (`AppModelInstallDescriptor`),
/// which this client has no relationship to.
public struct HuggingFaceModelSearchClient: Sendable {
    private let session: URLSession
    private let baseURL: URL

    public init(session: URLSession = .shared,
                baseURL: URL = URL(string: "https://huggingface.co/api/models")!) {
        self.session = session
        self.baseURL = baseURL
    }

    public func search(query: String, limit: Int = 20) async throws -> [HuggingFaceModelSummary] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "search", value: trimmed),
            URLQueryItem(name: "limit", value: String(limit)),
            URLQueryItem(name: "sort", value: "downloads"),
            URLQueryItem(name: "direction", value: "-1"),
        ]
        guard let url = components.url else { throw HuggingFaceModelSearchError.invalidResponse }

        let (data, response) = try await session.data(from: url)
        guard let http = response as? HTTPURLResponse else {
            throw HuggingFaceModelSearchError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            throw HuggingFaceModelSearchError.httpStatus(http.statusCode)
        }

        do {
            let decoded = try JSONDecoder().decode([HFAPIModel].self, from: data)
            return decoded.map { $0.summary }
        } catch {
            throw HuggingFaceModelSearchError.decodingFailed
        }
    }
}

/// Raw decode target for one entry of the Hub's `/api/models` response.
/// Kept private/internal to this file — callers only ever see
/// `HuggingFaceModelSummary`.
private struct HFAPIModel: Decodable {
    let id: String
    let downloads: Int?
    let likes: Int?
    let pipelineTag: String?
    let tags: [String]?

    enum CodingKeys: String, CodingKey {
        case id
        case downloads
        case likes
        case pipelineTag = "pipeline_tag"
        case tags
    }

    var summary: HuggingFaceModelSummary {
        HuggingFaceModelSummary(
            repoID: id,
            downloads: downloads ?? 0,
            likes: likes ?? 0,
            pipelineTag: pipelineTag,
            tags: tags ?? [])
    }
}
