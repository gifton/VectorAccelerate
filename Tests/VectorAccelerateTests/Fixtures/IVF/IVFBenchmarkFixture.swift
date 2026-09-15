import Foundation

/// CPU-owned input for future IVF adapters; no buffers, timing, or run schema.
struct IVFBenchmarkFixture: Sendable {
    let dimension: Int
    let centroids: [[Float]]
    let vectors: [[Float]] // CSR ordered
    let queries: [[Float]]
    let listOffsets: [UInt32]
    let originalIDs: [UInt32]
    let selectedLists: [[UInt32]]

    var selectedCandidateCounts: [Int] {
        selectedLists.map { lists in
            lists.reduce(0) { $0 + Int(listOffsets[Int($1) + 1] - listOffsets[Int($1)]) }
        }
    }
}

enum IVFBenchmarkFixtureError: Error { case invalidShape }

enum IVFBenchmarkFixtures {
    /// Controlled binary-grid clusters, not a claim about real embedding distributions.
    /// Streams s...s+3 are reserved for centers, vectors, queries, and original-ID order.
    /// Generation is independent of case execution order and does not mutate shared state.
    static func make(seed: UInt64, stream: UInt64 = 0, dimension: Int,
                     listSizes: [Int], queryCount: Int, nprobe: Int) throws -> IVFBenchmarkFixture {
        guard dimension > 0, dimension <= Int(UInt32.max), !listSizes.isEmpty,
              queryCount > 0, nprobe > 0, nprobe <= listSizes.count,
              listSizes.allSatisfy({ $0 >= 0 }) else { throw IVFBenchmarkFixtureError.invalidShape }
        var offsets: [UInt32] = [0]
        var count = 0
        for size in listSizes {
            let sum = count.addingReportingOverflow(size)
            guard !sum.overflow, sum.partialValue < Int(UInt32.max) else { throw IVFBenchmarkFixtureError.invalidShape }
            count = sum.partialValue
            offsets.append(UInt32(count))
        }
        for rows in [count, listSizes.count, queryCount] {
            let elements = rows.multipliedReportingOverflow(by: dimension)
            guard !elements.overflow, elements.partialValue <= Int.max / MemoryLayout<Float>.stride else {
                throw IVFBenchmarkFixtureError.invalidShape
            }
        }
        // Binary fractions keep reference distances exactly representable for the
        // supported benchmark dimensions, without duplicating the RNG algorithm.
        func grid(_ rows: [[Float]]) -> [[Float]] {
            rows.map { $0.map { ($0 * 16).rounded(.down) / 16 } }
        }
        var centersGenerator = TestDataGenerator(seed: seed, stream: stream)
        var vectorsGenerator = TestDataGenerator(seed: seed, stream: stream &+ 1)
        var queriesGenerator = TestDataGenerator(seed: seed, stream: stream &+ 2)
        var idGenerator = TestRNG(seed: seed, stream: stream &+ 3)
        let centers = grid(centersGenerator.uniformVectors(count: listSizes.count, dimension: dimension))
        let noise = grid(vectorsGenerator.uniformVectors(count: count, dimension: dimension, range: -0.25...0.25))
        var vectors: [[Float]] = []
        vectors.reserveCapacity(count)
        for list in listSizes.indices {
            for row in Int(offsets[list])..<Int(offsets[list + 1]) {
                vectors.append(zip(centers[list], noise[row]).map(+))
            }
        }
        let queries = grid(queriesGenerator.uniformVectors(count: queryCount, dimension: dimension))
        var ids = (0..<count).map(UInt32.init)
        idGenerator.shuffle(&ids)
        let selected = queries.map { query in
            centers.indices.sorted { a, b in
                let da = zip(query, centers[a]).reduce(0.0) { $0 + pow(Double($1.0) - Double($1.1), 2) }
                let db = zip(query, centers[b]).reduce(0.0) { $0 + pow(Double($1.0) - Double($1.1), 2) }
                return da == db ? a < b : da < db
            }.prefix(nprobe).map(UInt32.init)
        }
        return IVFBenchmarkFixture(dimension: dimension, centroids: centers, vectors: vectors,
            queries: queries, listOffsets: offsets, originalIDs: ids, selectedLists: selected)
    }
}
