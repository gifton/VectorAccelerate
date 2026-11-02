import XCTest

final class IVFBenchmarkFixtureTests: XCTestCase {
    func testPreparationBuildsCSRAndResetsStreamsPerCall() throws {
        let fixture = try IVFBenchmarkFixtures.make(seed: 42, dimension: 4,
            listSizes: [2, 0, 3], queryCount: 2, nprobe: 2)
        XCTAssertEqual(fixture.listOffsets, [0, 2, 2, 5])
        XCTAssertEqual(fixture.vectors.count, 5)
        XCTAssertEqual(fixture.centroids.count, 3)
        XCTAssertEqual(fixture.queries.count, 2)
        XCTAssertEqual(fixture.originalIDs.sorted(), [0, 1, 2, 3, 4])
        XCTAssertEqual(fixture.selectedLists.map(\.count), [2, 2])
    }

    func testGenerationIsIndependentOfCallOrderAndQueryCount() throws {
        let first = try IVFBenchmarkFixtures.make(seed: 42, stream: 7, dimension: 8,
            listSizes: [65, 0, 257], queryCount: 3, nprobe: 2)
        _ = try IVFBenchmarkFixtures.make(seed: 99, dimension: 128,
            listSizes: [1, 8], queryCount: 1, nprobe: 1)
        let repeated = try IVFBenchmarkFixtures.make(seed: 42, stream: 7, dimension: 8,
            listSizes: [65, 0, 257], queryCount: 3, nprobe: 2)
        XCTAssertEqual(first.centroids, repeated.centroids)
        XCTAssertEqual(first.vectors, repeated.vectors)
        XCTAssertEqual(first.queries, repeated.queries)
        XCTAssertEqual(first.originalIDs, repeated.originalIDs)
        XCTAssertEqual(first.selectedLists, repeated.selectedLists)
        let largerBatch = try IVFBenchmarkFixtures.make(seed: 42, stream: 7, dimension: 8,
            listSizes: [65, 0, 257], queryCount: 8, nprobe: 2)
        XCTAssertEqual(first.queries, Array(largerBatch.queries.prefix(3)))
        XCTAssertEqual(first.selectedLists, Array(largerBatch.selectedLists.prefix(3)))
        XCTAssertEqual(first.vectors, largerBatch.vectors)
        for (lists, actual) in zip(first.selectedLists, first.selectedCandidateCounts) {
            XCTAssertEqual(actual, lists.reduce(0) { $0 + [65, 0, 257][Int($1)] })
        }
    }

    // Captured from the existing version-one utilities. Promotion must preserve
    // these bits; numerical oracle correctness is checked separately with literals.
    func testPreparedFixtureGoldenForGeneratorVersionOne() throws {
        let fixture = try IVFBenchmarkFixtures.make(seed: 42, dimension: 4,
            listSizes: [2, 0, 3], queryCount: 2, nprobe: 2)
        XCTAssertEqual(fixture.centroids.flatMap { $0 }.map(\.bitPattern), [
            0xBF500000, 0xBE400000, 0x3F000000, 0xBEE00000,
            0xBF800000, 0x3F100000, 0x3EA00000, 0xBE000000,
            0x3F300000, 0x3E400000, 0x3F200000, 0x3F400000
        ])
        XCTAssertEqual(fixture.vectors.flatMap { $0 }.map(\.bitPattern), [
            0xBF400000, 0xBEE00000, 0x3E800000, 0xBF200000,
            0xBF300000, 0xBE400000, 0x3EE00000, 0xBF300000,
            0x3F500000, 0x3E000000, 0x3F300000, 0x3F200000,
            0x3F100000, 0xBD800000, 0x3F200000, 0x3F200000,
            0x3EE00000, 0x3EC00000, 0x3F200000, 0x3F100000
        ])
        XCTAssertEqual(fixture.queries.flatMap { $0 }.map(\.bitPattern), [
            0xBF100000, 0x3F100000, 0xBE800000, 0xBF800000,
            0x3E400000, 0xBEA00000, 0xBF100000, 0x3EA00000
        ])
        XCTAssertEqual(fixture.originalIDs, [4, 0, 1, 2, 3])
        XCTAssertEqual(fixture.selectedLists, [[1, 0], [2, 0]])
    }

    func testMalformedPreparationIsRejectedAndEmptyListsRemainUsable() throws {
        for sizes in [[-1], [Int.max], []] {
            XCTAssertThrowsError(try IVFBenchmarkFixtures.make(seed: 1, dimension: 4,
                listSizes: sizes, queryCount: 1, nprobe: 1))
        }
        for dimension in [0, Int.max] {
            XCTAssertThrowsError(try IVFBenchmarkFixtures.make(seed: 1, dimension: dimension,
                listSizes: [1], queryCount: 1, nprobe: 1))
        }
        XCTAssertThrowsError(try IVFBenchmarkFixtures.make(seed: 1, dimension: 4,
            listSizes: [1], queryCount: 1, nprobe: 2))
        let empty = try IVFBenchmarkFixtures.make(seed: 1, dimension: 4,
            listSizes: [0, 0], queryCount: 2, nprobe: 2)
        XCTAssertEqual(empty.listOffsets, [0, 0, 0])
        XCTAssertEqual(empty.selectedCandidateCounts, [0, 0])
        XCTAssertEqual(try IVFBenchmarkOracle.exact(empty, query: 0, k: 8), [])
        XCTAssertEqual(try IVFBenchmarkOracle.retained(empty, query: 0, k: 8), [])
        XCTAssertThrowsError(try IVFBenchmarkOracle.exact(empty, query: 2, k: 8))
        XCTAssertThrowsError(try IVFBenchmarkOracle.retained(empty, query: 0, k: 8, width: 0))
    }

    func testExactOracleOrdersOriginalIDsAndAppliesEligibility() throws {
        let fixture = IVFBenchmarkFixture(dimension: 1, centroids: [[0], [10]],
            vectors: [[0], [3], [1], [2], [.nan], [.infinity]], queries: [[0]],
            listOffsets: [0, 3, 6], originalIDs: [50, 30, 10, 20, 40, 60], selectedLists: [[1, 0]])
        let exact = try IVFBenchmarkOracle.exact(fixture, query: 0, k: 8)
        XCTAssertEqual(exact.map(\.index), [50, 10, 20, 30, 60, 40])
        XCTAssertEqual(Array(exact.prefix(5).map(\.distance)), [0, 1, 4, 9, .infinity])
        XCTAssertTrue(exact.last?.distance.isNaN == true)
        let filtered = try IVFBenchmarkOracle.exact(fixture, query: 0, k: 2, eligibleIDs: [20, 30, 40])
        XCTAssertEqual(filtered.map(\.index), [20, 30])
        XCTAssertEqual(filtered.map(\.distance), [4, 9])
    }

    func testRetainedOracleRestartsEachListAndDoesNotClaimGlobalExactness() throws {
        var rows = [[Float]](repeating: [1000], count: 258)
        rows[0] = [-1]
        for i in 0..<9 { rows[1 + i * 32] = [Float(i)] }
        let fixture = IVFBenchmarkFixture(dimension: 1, centroids: [[0], [0]],
            vectors: rows, queries: [[0]], listOffsets: [0, 1, 258],
            originalIDs: (0..<258).map(UInt32.init), selectedLists: [[0, 1]])
        let retained = try IVFBenchmarkOracle.retained(fixture, query: 0, k: 9, width: 32)
        XCTAssertEqual(retained.map(\.index), [1, 0, 33, 65, 97, 129, 161, 193, 2])
        XCTAssertEqual(retained.map(\.distance), [0, 1, 1, 4, 9, 16, 25, 36, 1_000_000])
        let exact = try IVFBenchmarkOracle.exact(fixture, query: 0, k: 9)
        XCTAssertEqual(exact.map(\.index), [1, 0, 33, 65, 97, 129, 161, 193, 225])
    }
}
