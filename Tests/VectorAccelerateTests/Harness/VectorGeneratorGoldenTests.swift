import XCTest
import VectorTestKit

/// FROZEN per-method digests of VectorGenerator at seed 42 / stream 0 (spec §4.3). Generated once from the
/// pre-promotion TestDataGenerator on 2026-09-18. If a method's output must change, bump VectorGenerator.version
/// and update the single affected digest in the same commit with a ledger entry. Do not regenerate wholesale.
/// These digests are the `-Onone` (debug) stream, which is the contract by ruling (Task 3 finding: `-O` fuses an
/// adjacent `sinf`/`cosf` pair into `__sincosf_stret`, shifting Gaussian values by 1-2 ulp); regenerating with
/// `swiftc -O` against a pre-wrapper `TestRNG.swift` produces a different, wrong file.
final class VectorGeneratorGoldenTests: XCTestCase {
    private func d(_ v: [[Float]]) -> String { ContentDigest.sha256([.int(v.count), .int(v.first?.count ?? 0), .floats(v.flatMap { $0 })]) }

    func testVersionIsOne() { XCTAssertEqual(VectorGenerator.version, "1") }

    func testFrozenDigests() {
        var g = VectorGenerator(seed: 42, stream: 0)
        XCTAssertEqual(d(g.uniformVectors(count: 16, dimension: 8)), "f2091c1c91e42c57424e51bf238f1f100bb8d98ef11c9fc54a2574dd5450c360", "uniformVectors")
        XCTAssertEqual(d(g.uniformVectors(count: 16, dimension: 8, range: -3...3)), "9041c04aca00746c7fd4f1e86d30b6e08d12c77ee465063ed94ff517973e42be", "uniformVectors(range:)")
        XCTAssertEqual(d(g.gaussianClusters(numClusters: 4, pointsPerCluster: 8, dimension: 8)), "718b42ee7285b303a0da43a4665a47b30ed27ea9cb7de5d8783f7f3205585aee", "gaussianClusters")
        XCTAssertEqual(d(g.separatedGaussianClusters(numClusters: 4, pointsPerCluster: 8, dimension: 8).data), "833af90da5e61363efb50f507cb97d2c5e0c78d90296e86ddec53037aa68f9be", "separatedGaussianClusters")
        XCTAssertEqual(d(g.unitVectors(count: 16, dimension: 8)), "b5b190fa3fc37b25718c750f134147962f6e37f13b0de14e4879f644364dd9b3", "unitVectors")
        XCTAssertEqual(d(g.clusteredUnitVectors(numClusters: 4, pointsPerCluster: 8, dimension: 8)), "6168cbe41f830f1b9ebc4c7d4e376c5dfd5647f898f2f9498b9346f538d9214d", "clusteredUnitVectors")
        XCTAssertEqual(d(g.sparseVectors(count: 16, dimension: 8)), "f3903fab13f098a36455fd203ca3ad5ab784f2449355d0d4ac620c987dfaf8ed", "sparseVectors")
        XCTAssertEqual(d(g.skewedClusters(numClusters: 4, totalPoints: 32, dimension: 8)), "ed553cc04bb296129b862f6dce8e9eebe9c6b63c5e65cb6d21ce883c19a7452c", "skewedClusters")
        let base = g.uniformVectors(count: 8, dimension: 4)
        XCTAssertEqual(d(g.perturbedQueries(from: base, count: 4)), "19fc1ba311ebcf7f40321c248d3c19284c17bddde104b1b771dadc7b41bef3f8", "perturbedQueries")
        XCTAssertEqual(d(g.clusterQueries(centers: base, countPerCenter: 1)), "d05ab8811e13e19f3a02c8fe9024ea12853c5941bc24d1563d97364f31592577", "clusterQueries")
        XCTAssertEqual(d(g.randomQueries(count: 4, dimension: 4)), "6db07082ed1ca00ecfac5323b11c8d379b66c1bb98926d03e47bbc1415b7af45", "randomQueries")
    }
}
