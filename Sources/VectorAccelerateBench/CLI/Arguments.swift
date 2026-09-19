import Foundation

/// Hand-written parser: `<command> [<subcommand>] [--key value | --flag] ...`. No third-party dependency (spec §5.5).
struct Arguments {
    static let booleanFlags: Set<String> = ["--approve", "--allow-dirty", "--strict-perf", "--require-gpu", "--validation-layers", "--help", "-h"]

    let command: String
    let subcommand: String?
    let options: [String: String]
    let flags: Set<String>

    init(_ argv: [String]) {
        var positionals: [String] = [], options: [String: String] = [:], flags: Set<String> = []
        var i = 0
        while i < argv.count {
            let a = argv[i]
            if a.hasPrefix("--") || a == "-h" {
                if Self.booleanFlags.contains(a) { flags.insert(a); i += 1; continue }
                if i + 1 < argv.count { options[a] = argv[i + 1]; i += 2 } else { options[a] = ""; i += 1 }
            } else { positionals.append(a); i += 1 }
        }
        command = positionals.first ?? "help"
        subcommand = positionals.count > 1 ? positionals[1] : nil
        self.options = options; self.flags = flags
    }

    func value(_ key: String) -> String? { options[key] }
    func has(_ flag: String) -> Bool { flags.contains(flag) }

    /// An absent option is `nil` (the caller supplies the default); a present value that `parse` rejects throws `BenchError.usage`
    /// naming the option and the value.
    func parsed<T>(_ key: String, expecting what: String, _ parse: (String) -> T?) throws -> T? {
        guard let raw = options[key] else { return nil }
        guard let parsedValue = parse(raw) else { throw BenchError.usage("\(key) expects \(what), got '\(raw)'") }
        return parsedValue
    }
}
