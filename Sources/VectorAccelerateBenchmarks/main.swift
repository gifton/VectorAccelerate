//
//  main.swift
//  VectorAccelerateBenchmarks
//
//  Simple benchmark runner for VectorAccelerate performance testing.
//  Uses Metal 4 exclusively (iOS 26+, macOS 26+).
//

import Foundation
import VectorAccelerate
import VectorCore

struct BenchmarkRunner {
    static func main() async throws {
        print("🚀 VectorAccelerate Benchmark Suite (Metal 4)")
        print("=============================================\n")

        // Check Metal availability using VectorCore's ComputeDevice
        guard ComputeDevice.gpu().isAvailable else {
            print("❌ Metal is not available on this system")
            return
        }

        print("✅ Metal 4 is available")

        do {
            // Initialize Metal 4 context and compute engine
            let context = try await Metal4Context()
            let engine = try await Metal4ComputeEngine(context: context)

            print("✅ Metal 4 context and compute engine initialized")
            print("📊 Starting benchmarks...\n")

            // Create benchmark suite
            let suite = BenchmarkSuite(engine: engine, context: context)

            // Define benchmark configurations
            // Includes 384, 512, 768, 1536 to test optimized kernels (VectorCore 0.1.5)
            let configurations = [
                BenchmarkConfiguration(
                    iterations: 50,
                    warmupIterations: 5,
                    dimension: 128,
                    batchSize: 100,
                    additionalInfo: ["description": "Base dimension (generic kernel)"]
                ),
                BenchmarkConfiguration(
                    iterations: 50,
                    warmupIterations: 5,
                    dimension: 384,
                    batchSize: 100,
                    additionalInfo: ["description": "MiniLM/Sentence-BERT (optimized kernel)"]
                ),
                BenchmarkConfiguration(
                    iterations: 50,
                    warmupIterations: 5,
                    dimension: 512,
                    batchSize: 100,
                    additionalInfo: ["description": "OpenAI Ada (optimized kernel)"]
                ),
                BenchmarkConfiguration(
                    iterations: 50,
                    warmupIterations: 5,
                    dimension: 768,
                    batchSize: 100,
                    additionalInfo: ["description": "BERT-base (optimized kernel)"]
                ),
                BenchmarkConfiguration(
                    iterations: 25,
                    warmupIterations: 3,
                    dimension: 1536,
                    batchSize: 100,
                    additionalInfo: ["description": "OpenAI text-embedding-3-large (optimized kernel)"]
                )
            ]

            // Run benchmarks
            print("Running benchmarks for dimensions: 128, 384, 512, 768, 1536")
            print("(384, 512, 768, 1536 use optimized kernels)\n")
            let results = try await suite.runAllBenchmarks(configurations: configurations)

            // Generate and print report
            let report = BenchmarkReport.consoleReport(results: results)
            print(report)

            // Save JSON report
            let jsonData = try BenchmarkReport.jsonReport(results: results)
            let url = URL(fileURLWithPath: "benchmark_results.json")
            try jsonData.write(to: url)
            print("📄 Detailed results saved to: \(url.path)")

            // Print summary
            printSummary(results: results)

        } catch {
            print("❌ Benchmark failed: \(error)")
            throw error
        }
    }

    private static func printSummary(results: [VectorAccelerate.BenchmarkResult]) {
        print("\n🏆 Performance Summary")
        print("=====================")

        // Group by operation type
        let groupedResults = Dictionary(grouping: results) { result in
            result.name.components(separatedBy: " ").first ?? "Unknown"
        }

        for (operation, operationResults) in groupedResults.sorted(by: { $0.key < $1.key }) {
            print("\n\(operation):")

            for result in operationResults.sorted(by: { $0.configuration.dimension < $1.configuration.dimension }) {
                let timeMs = result.medianTime * 1000
                let opsPerSec = result.operationsPerSecond

                print("  dim \(result.configuration.dimension): \(String(format: "%.2f", timeMs))ms (\(String(format: "%.0f", opsPerSec)) ops/sec)")

                if let vectorsPerSec = result.vectorsPerSecond {
                    print("    ↳ \(String(format: "%.0f", vectorsPerSec)) vectors/sec")
                }
            }
        }

        // Find fastest operations
        let fastestResult = results.min(by: { $0.medianTime < $1.medianTime })
        if let fastest = fastestResult {
            print("\n⚡ Fastest Operation: \(fastest.name)")
            print("   Time: \(String(format: "%.3f", fastest.medianTime * 1000))ms")
        }

        // Calculate average performance across dimensions
        let dimensionGroups = Dictionary(grouping: results) { $0.configuration.dimension }
        print("\n📈 Performance by Dimension:")

        for dimension in dimensionGroups.keys.sorted() {
            let dimensionResults = dimensionGroups[dimension]!
            let avgTime = dimensionResults.map(\.medianTime).reduce(0, +) / Double(dimensionResults.count)
            print("  \(dimension)D: \(String(format: "%.3f", avgTime * 1000))ms average")
        }
    }
}

// MARK: - Entry Point

// Top-level async entry point
let semaphore = DispatchSemaphore(value: 0)
let args = CommandLine.arguments

Task {
    do {
        if args.contains("--index") || args.contains("-i") {
            // Run the new index benchmark harness
            try await IndexBenchmarkRunner.run()
        } else {
            // Run the original kernel benchmarks
            try await BenchmarkRunner.main()
        }
    } catch {
        print("❌ Fatal error: \(error)")
    }
    semaphore.signal()
}
semaphore.wait()
