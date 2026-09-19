import Foundation

let usage = """
VectorAccelerateBench <command>
  list [--filter <glob>] [--exclude <glob>]
  run  [--filter <glob>] [--exclude <glob>] [--mode smoke|quick|full] [--samples N] [--iterations N] [--seed S]
       [--out <path>] [--label <text>] [--require-gpu] [--validation-layers] [--probe-slowdown N]
  baseline capture --from <artifact> [--approve] [--allow-dirty]
  compare --artifact <path> [--against approved|<n>] [--strict-perf] [--format text|json|markdown]
  validate-artifact <path>
"""

let args = Arguments(Array(CommandLine.arguments.dropFirst()))
if args.has("--help") || args.has("-h") { print(usage); exit(0) }
var status: Int32 = 2
do {
    switch args.command {
    case "list": status = try ListCommand.run(args)
    case "validate-artifact": status = try ValidateCommand.run(args)
    case "run": status = try await RunCommand.run(args)              // Task 15
    case "baseline": status = try BaselineCommand.run(args)           // Task 16
    case "compare": status = try CompareCommand.run(args)             // Task 16
    default: print(usage); status = args.command == "help" ? 0 : 2
    }
} catch let error as BenchError {
    Diagnostics.line("error: \(error)")
    status = error.exitCode
} catch {
    Diagnostics.line("error: \(error)")
    status = 1
}
exit(status)
