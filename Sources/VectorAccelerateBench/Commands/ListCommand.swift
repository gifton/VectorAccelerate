import Foundation
import VectorTestKit

enum ListCommand {
    static func run(_ args: Arguments) throws -> Int32 {
        try args.validate(options: ["--filter", "--exclude"], flags: [], operands: 0)
        let cases = try Registry.select(include: args.value("--filter").map { [$0] } ?? [], exclude: args.value("--exclude").map { [$0] } ?? [])
        for c in cases { print("\(c.id.rawValue)  route=\(c.identity.route)  timing=\(c.providedTiming.map(\.rawValue).sorted().joined(separator: ","))") }
        print("\(cases.count) case(s)")
        return 0
    }
}
