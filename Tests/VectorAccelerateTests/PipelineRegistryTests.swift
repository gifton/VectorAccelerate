//
//  PipelineRegistryTests.swift
//  VectorAccelerateTests
//
//  Tests for PipelineRegistry pipeline tier categorization
//

import XCTest
@testable import VectorAccelerate

final class PipelineRegistryTests: XCTestCase {

    // MARK: - Initialization Tests

    func testDefaultRegistry() {
        let registry = PipelineRegistry.default

        // Default registry should have common keys as critical
        XCTAssertFalse(registry.criticalKeys.isEmpty)
        XCTAssertTrue(registry.occasionalKeys.isEmpty)
        XCTAssertTrue(registry.rareKeys.isEmpty)
    }

    func testJournalingAppRegistry() {
        let registry = PipelineRegistry.journalingApp

        // Journaling app should have specific critical keys
        XCTAssertFalse(registry.criticalKeys.isEmpty)
        XCTAssertFalse(registry.occasionalKeys.isEmpty)
        XCTAssertFalse(registry.rareKeys.isEmpty)

        // Critical should include 384-dim L2 distance (MiniLM)
        let hasL2_384 = registry.criticalKeys.contains { $0.operation == "l2Distance" && $0.dimension == 384 }
        XCTAssertTrue(hasL2_384, "Journaling app should have l2Distance(384) as critical")

        // Critical should include cosine similarity for 384
        let hasCosine_384 = registry.criticalKeys.contains { $0.operation == "cosineSimilarity" && $0.dimension == 384 }
        XCTAssertTrue(hasCosine_384, "Journaling app should have cosineSimilarity(384) as critical")
    }

    func testMinimalRegistry() {
        let registry = PipelineRegistry.minimal

        // Minimal should have only one critical key
        XCTAssertEqual(registry.criticalKeys.count, 1)
        XCTAssertTrue(registry.occasionalKeys.isEmpty)
        XCTAssertTrue(registry.rareKeys.isEmpty)
    }

    func testCustomRegistry() {
        let customKey = PipelineCacheKey(operation: "custom_kernel")
        let registry = PipelineRegistry(entries: [
            .critical: [customKey],
            .occasional: [],
            .rare: []
        ])

        XCTAssertEqual(registry.criticalKeys.count, 1)
        XCTAssertEqual(registry.criticalKeys.first?.operation, "custom_kernel")
    }

    // MARK: - Tier Lookup Tests

    func testTierLookupKnownKey() {
        let registry = PipelineRegistry.journalingApp

        let tier = registry.tier(for: .l2Distance(dimension: 384))
        XCTAssertEqual(tier, .critical)
    }

    func testTierLookupOccasionalKey() {
        let registry = PipelineRegistry.journalingApp

        // 768-dim should be occasional
        let tier = registry.tier(for: .l2Distance(dimension: 768))
        XCTAssertEqual(tier, .occasional)
    }

    func testTierLookupUnknownKey() {
        let registry = PipelineRegistry.journalingApp

        // Unknown key should default to rare
        let unknownKey = PipelineCacheKey(operation: "unknown_operation")
        let tier = registry.tier(for: unknownKey)
        XCTAssertEqual(tier, .rare)
    }

    // MARK: - Keys Access Tests

    func testKeysForTier() {
        let registry = PipelineRegistry.journalingApp

        let criticalKeys = registry.keys(for: .critical)
        let occasionalKeys = registry.keys(for: .occasional)
        let rareKeys = registry.keys(for: .rare)

        XCTAssertFalse(criticalKeys.isEmpty)
        XCTAssertFalse(occasionalKeys.isEmpty)
        XCTAssertFalse(rareKeys.isEmpty)
    }

    func testTotalCount() {
        let registry = PipelineRegistry.journalingApp

        let total = registry.totalCount
        let sumOfTiers = registry.criticalKeys.count + registry.occasionalKeys.count + registry.rareKeys.count

        XCTAssertEqual(total, sumOfTiers)
    }

    func testCountForTier() {
        let registry = PipelineRegistry(entries: [
            .critical: [.l2Distance(dimension: 384), .cosineSimilarity(dimension: 384)],
            .occasional: [.dotProduct(dimension: 0)],
            .rare: []
        ])

        XCTAssertEqual(registry.count(for: .critical), 2)
        XCTAssertEqual(registry.count(for: .occasional), 1)
        XCTAssertEqual(registry.count(for: .rare), 0)
    }

    // MARK: - PipelineTier Tests

    func testPipelineTierAllCases() {
        let allCases = PipelineTier.allCases

        XCTAssertEqual(allCases.count, 3)
        XCTAssertTrue(allCases.contains(.critical))
        XCTAssertTrue(allCases.contains(.occasional))
        XCTAssertTrue(allCases.contains(.rare))
    }

    func testPipelineTierRawValue() {
        XCTAssertEqual(PipelineTier.critical.rawValue, "critical")
        XCTAssertEqual(PipelineTier.occasional.rawValue, "occasional")
        XCTAssertEqual(PipelineTier.rare.rawValue, "rare")
    }

    // MARK: - Codable Tests

    func testCodableRoundTrip() throws {
        let original = PipelineRegistry(entries: [
            .critical: [.l2Distance(dimension: 384)],
            .occasional: [.cosineSimilarity(dimension: 0)],
            .rare: []
        ])

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(PipelineRegistry.self, from: data)

        XCTAssertEqual(original.criticalKeys.count, decoded.criticalKeys.count)
        XCTAssertEqual(original.occasionalKeys.count, decoded.occasionalKeys.count)
        XCTAssertEqual(original.rareKeys.count, decoded.rareKeys.count)
    }

    // MARK: - EmbeddingFocused Registry Tests

    func testEmbeddingFocusedRegistry() {
        let registry = PipelineRegistry.embeddingFocused

        // Should include embedding model keys
        XCTAssertFalse(registry.criticalKeys.isEmpty)

        // Should include top-K and normalization
        let hasTopK = registry.criticalKeys.contains { $0.operation == "topK" }
        XCTAssertTrue(hasTopK, "Embedding focused should have topK as critical")
    }
}
