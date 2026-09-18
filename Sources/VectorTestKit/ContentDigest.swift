import Foundation
import CryptoKit

/// One component of a fixture's identity. Arrays are prefixed by their element count.
public enum DigestPart: Sendable {
    case floats([Float])
    case doubles([Double])
    case int32s([Int32])
    case uint32s([UInt32])
    case int64s([Int64])
    case uint64s([UInt64])
    case int(Int)
    case double(Double)
    case string(String)
}

/// SHA-256 over a canonical little-endian encoding (spec §4.4). Rendered as lowercase hex.
public enum ContentDigest {
    public static func sha256(_ parts: [DigestPart]) -> String {
        var hasher = SHA256()
        for part in parts { append(part, to: &hasher) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func append(_ part: DigestPart, to hasher: inout SHA256) {
        switch part {
        case .floats(let a):  appendCount(a.count, &hasher); for x in a { appendLE(x.bitPattern, &hasher) }
        case .doubles(let a): appendCount(a.count, &hasher); for x in a { appendLE(x.bitPattern, &hasher) }
        case .int32s(let a):  appendCount(a.count, &hasher); for x in a { appendLE(UInt32(bitPattern: x), &hasher) }
        case .uint32s(let a): appendCount(a.count, &hasher); for x in a { appendLE(x, &hasher) }
        case .int64s(let a):  appendCount(a.count, &hasher); for x in a { appendLE(UInt64(bitPattern: x), &hasher) }
        case .uint64s(let a): appendCount(a.count, &hasher); for x in a { appendLE(x, &hasher) }
        case .int(let v):     appendLE(UInt64(bitPattern: Int64(v)), &hasher)
        case .double(let v):  appendLE(v.bitPattern, &hasher)
        case .string(let s):
            let bytes = Array(s.utf8)
            appendCount(bytes.count, &hasher)
            hasher.update(data: Data(bytes))
        }
    }

    private static func appendCount(_ n: Int, _ hasher: inout SHA256) {
        appendLE(UInt64(n), &hasher)
    }

    private static func appendLE<T: FixedWidthInteger>(_ value: T, _ hasher: inout SHA256) {
        var le = value.littleEndian
        withUnsafeBytes(of: &le) { hasher.update(bufferPointer: $0) }
    }
}
