import Foundation

struct IVFReferenceNeighbor: Equatable, Sendable {
    let index: UInt32
    let distance: Double
}

enum IVFBenchmarkOracle {
    /// All eligible rows in the original corpus, regardless of selected lists.
    /// Returns real neighbors only; the caller checks output padding separately.
    static func exact(_ fixture: IVFBenchmarkFixture, query: Int, k: Int,
                      eligibleIDs: Set<UInt32>? = nil) throws -> [IVFReferenceNeighbor] {
        try validate(fixture, query: query, k: k)
        let rows = fixture.vectors.indices.filter { eligibleIDs?.contains(fixture.originalIDs[$0]) ?? true }
        return Array(rows.map { neighbor(fixture, query: query, row: $0) }.sorted(by: precedes).prefix(k))
    }

    /// Independent sorting oracle for the current eight-candidates-per-lane contract.
    /// It does not emulate the shader's insertion heap, SIMD reduction, or bitonic sort.
    static func retained(_ fixture: IVFBenchmarkFixture, query: Int, k: Int,
                         width: Int = 256) throws -> [IVFReferenceNeighbor] {
        try validate(fixture, query: query, k: k)
        guard width > 0, width <= 256, width.isMultiple(of: 32) else { throw IVFBenchmarkFixtureError.invalidShape }
        var lanes = [[IVFReferenceNeighbor]](repeating: [], count: width)
        for list in fixture.selectedLists[query] {
            let start = Int(fixture.listOffsets[Int(list)])
            let end = Int(fixture.listOffsets[Int(list) + 1])
            for row in start..<end {
                lanes[(row - start) % width].append(neighbor(fixture, query: query, row: row))
            }
        }
        let pool = lanes.flatMap { $0.sorted(by: precedes).prefix(8) }
        return Array(pool.sorted(by: precedes).prefix(k))
    }

    private static func neighbor(_ fixture: IVFBenchmarkFixture, query: Int, row: Int) -> IVFReferenceNeighbor {
        let squared = zip(fixture.queries[query], fixture.vectors[row]).reduce(0.0) { sum, pair in
            let delta = Double(pair.0) - Double(pair.1)
            return sum + delta * delta
        }
        return IVFReferenceNeighbor(index: fixture.originalIDs[row], distance: squared)
    }

    private static func precedes(_ a: IVFReferenceNeighbor, _ b: IVFReferenceNeighbor) -> Bool {
        if a.distance.isNaN != b.distance.isNaN { return !a.distance.isNaN }
        if !a.distance.isNaN && a.distance != b.distance { return a.distance < b.distance }
        return a.index < b.index
    }

    private static func validate(_ fixture: IVFBenchmarkFixture, query: Int, k: Int) throws {
        guard k >= 0, fixture.queries.indices.contains(query), fixture.dimension > 0,
              fixture.queries.allSatisfy({ $0.count == fixture.dimension }),
              fixture.vectors.allSatisfy({ $0.count == fixture.dimension }),
              fixture.centroids.allSatisfy({ $0.count == fixture.dimension }),
              fixture.originalIDs.count == fixture.vectors.count,
              Set(fixture.originalIDs).count == fixture.originalIDs.count,
              !fixture.originalIDs.contains(.max),
              fixture.listOffsets.count == fixture.centroids.count + 1,
              fixture.listOffsets.first == 0,
              fixture.listOffsets.last == UInt32(exactly: fixture.vectors.count),
              zip(fixture.listOffsets, fixture.listOffsets.dropFirst()).allSatisfy({ $0 <= $1 }),
              fixture.selectedLists.count == fixture.queries.count,
              fixture.selectedLists.allSatisfy({ lists in
                  Set(lists).count == lists.count && lists.allSatisfy { $0 < fixture.centroids.count }
              }) else { throw IVFBenchmarkFixtureError.invalidShape }
    }
}
