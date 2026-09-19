import Foundation

let usage = """
VectorAccelerateBench <command>
  list [--filter <glob>] [--exclude <glob>]
  run  [--filter <glob>] [--exclude <glob>] [--mode smoke|quick|full] [--samples N] [--iterations N] [--seed S]
       [--out <path>] [--label <text>] [--require-gpu] [--validation-layers] [--allow-dirty] [--probe-slowdown N]
  baseline capture --from <artifact> [--approve] [--allow-dirty]
  compare --artifact <path> [--against approved|<n>] [--strict-perf] [--format text|json|markdown]
  validate-artifact <path>
"""

let args = Arguments(Array(CommandLine.arguments.dropFirst()))
var status: Int32 = 2
do {
    switch args.command {
    case "list": status = try ListCommand.run(args)
    case "validate-artifact": status = ValidateCommand.run(args)
    case "run": status = try await RunCommand.run(args)              // Task 15
    case "baseline": status = try BaselineCommand.run(args)           // Task 16
    case "compare": status = try CompareCommand.run(args)             // Task 16
    default: print(usage); status = args.command == "help" ? 0 : 2
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    status = 1
}
exit(status)

// Temporary stubs so the executable links before Tasks 15–16 land; each is deleted when its real file arrives.
enum RunCommand { static func run(_ args: Arguments) async throws -> Int32 { print("run: not implemented until Task 15"); return 2 } }
enum BaselineCommand { static func run(_ args: Arguments) throws -> Int32 { print("baseline: not implemented until Task 16"); return 2 } }
enum CompareCommand { static func run(_ args: Arguments) throws -> Int32 { print("compare: not implemented until Task 16"); return 2 } }
