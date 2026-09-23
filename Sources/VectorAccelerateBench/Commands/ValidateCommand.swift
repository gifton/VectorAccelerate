import Foundation
import VectorTestKit

enum ValidateCommand {
    static func run(_ args: Arguments) throws -> Int32 {
        try args.validate(options: [], flags: [], operands: 1)
        guard let path = args.subcommand else { throw BenchError.usage("usage: validate-artifact <path>") }
        do {
            let a = try RunStore.load(URL(fileURLWithPath: path))
            try a.validate()
            print("valid: schemaVersion \(a.schemaVersion), \(a.cases.count) case(s), key \(a.metadata.environmentKey)")
            return 0
        } catch {
            print("invalid: \(error)")
            return 1
        }
    }
}
