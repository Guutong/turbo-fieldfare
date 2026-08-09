import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite(.serialized)
struct HuggingFaceModelSearchClientTests {
    @Test func searchDecodesSummaries() async throws {
        let json = """
        [
            {
                "id": "mlx-community/gemma-4-26b-a4b-it-4bit",
                "downloads": 12345,
                "likes": 42,
                "pipeline_tag": "text-generation",
                "tags": ["mlx", "4-bit"]
            },
            {
                "id": "some-org/some-model",
                "downloads": 7,
                "likes": 0,
                "pipeline_tag": null,
                "tags": []
            }
        ]
        """.data(using: .utf8)!

        StubURLProtocol.handler = { request in
            #expect(request.url?.absoluteString.contains("search=gemma") == true)
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
            return (response, json)
        }

        let client = HuggingFaceModelSearchClient(session: Self.stubSession())
        let results = try await client.search(query: "gemma")

        #expect(results.count == 2)
        #expect(results[0].repoID == "mlx-community/gemma-4-26b-a4b-it-4bit")
        #expect(results[0].downloads == 12345)
        #expect(results[0].likes == 42)
        #expect(results[0].pipelineTag == "text-generation")
        #expect(results[1].pipelineTag == nil)
    }

    @Test func emptyQueryReturnsEmptyWithoutNetworkCall() async throws {
        StubURLProtocol.handler = { _ in
            Issue.record("network call should not happen for empty query")
            throw HuggingFaceModelSearchError.invalidResponse
        }
        let client = HuggingFaceModelSearchClient(session: Self.stubSession())
        let results = try await client.search(query: "   ")
        #expect(results.isEmpty)
    }

    @Test func nonSuccessStatusThrowsHTTPStatus() async throws {
        StubURLProtocol.handler = { request in
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 503, httpVersion: nil, headerFields: nil)!
            return (response, Data())
        }
        let client = HuggingFaceModelSearchClient(session: Self.stubSession())
        await #expect(throws: HuggingFaceModelSearchError.httpStatus(503)) {
            _ = try await client.search(query: "gemma")
        }
    }

    @Test func malformedJSONThrowsDecodingFailed() async throws {
        StubURLProtocol.handler = { request in
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
            return (response, Data("not json".utf8))
        }
        let client = HuggingFaceModelSearchClient(session: Self.stubSession())
        await #expect(throws: HuggingFaceModelSearchError.decodingFailed) {
            _ = try await client.search(query: "gemma")
        }
    }

    private static func stubSession() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        return URLSession(configuration: config)
    }
}

/// Minimal `URLProtocol` stub so tests never hit the network. Each test
/// installs its own `handler`; requests are answered synchronously from it.
private final class StubURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var handler: (@Sendable (URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let handler = StubURLProtocol.handler else {
            client?.urlProtocol(self, didFailWithError: HuggingFaceModelSearchError.invalidResponse)
            return
        }
        do {
            let (response, data) = try handler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}
