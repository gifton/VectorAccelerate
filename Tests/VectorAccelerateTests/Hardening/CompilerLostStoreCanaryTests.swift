//
//  CompilerLostStoreCanaryTests.swift
//  VectorAccelerateTests
//
//  Canary for the Swift optimizer miscompile VectorCore filed as VC-C001 (its T15 "lost store" class,
//  upstream swiftlang/swift#92562). At -O, SIL RedundantLoadElimination forwards the value an array literal
//  stored across a `withUnsafeMutableBufferPointer` call whose closure stays out of line and hands the buffer
//  to the writer `inout`, so a read after the call returns the literal although memory holds the write.
//  VectorCore's compiler investigation of 2026-10-30 (VectorCore-compiler-investigation-2026-10-30-evidence)
//  reproduced it with Apple Swift 6.4.2 and 6.2 at -O, -Osize and -O with WMO, showed that reading inside
//  the borrow does not help while the writer still takes the buffer `inout`, and showed that by-value
//  writers keep the store.
//
//  VectorAccelerate declares no `inout` unsafe-pointer parameter anywhere else (LostStoreShapeGuardTests
//  pins that), so the trigger shape exists only in this file, on purpose. Expectations:
//  - Debug (-Onone, RedundantLoadElimination does not run): the store is observed on every toolchain.
//  - Release on the compilers where the miscompile is demonstrated (6.2 <= compiler < 6.5): a lost store is
//    recorded as an expected, non-strict failure, so the gate stays green and the log states the toolchain's
//    condition; a toolchain that fixes it passes silently.
//  - Release on any other compiler: the store must be observed, and a lost store fails the gate so the
//    toolchain is re-qualified.
//  The controls pin the shapes VectorAccelerate actually relies on: by-value writers, a C-style out-pointer
//  call read inside the borrow (`PerformanceMonitor`), and accumulators written inside a read-only borrow
//  (the CPU fallbacks' sums). A control that fails on a future toolchain is a real alarm, not a canary.
//
//  Comparing this file's Release outcome with a build that passes
//  `-Xswiftc -Xllvm -Xswiftc -sil-disable-pass=redundant-load-elimination` separates compiler exposure from
//  a package regression, as the VectorCore report recommends. On 2026-10-30 the full VectorAccelerate
//  Release suite passed identically with and without that flag.
//

import XCTest

// MARK: - Shapes (each out of line, so the closure is not inlined into its caller)

@inline(never)
func lostStoreCanaryWriteOneInout(_ out: inout UnsafeMutableBufferPointer<Float>) {
    out[0] = 1
}

@inline(never)
func lostStoreCanaryWriteOneByValue(_ out: UnsafeMutableBufferPointer<Float>) {
    out[0] = 1
}

@inline(never)
func lostStoreCanaryWriteOneThroughPointer(_ out: UnsafeMutablePointer<Float>) {
    out.pointee = 1
}

/// The miscompiled shape: array literal, out-of-line closure that writes through the `inout` buffer, then a
/// direct element read after the call. Allowlisted in LostStoreShapeGuardTests.
@inline(never)
func lostStoreCanaryObservedAfterBorrow() -> Float {
    var values: [Float] = [2]
    values.withUnsafeMutableBufferPointer(lostStoreCanaryWriteOneInout)
    return values[0]
}

/// The inside-borrow variant VectorCore's investigation also found miscompiled: the read moves inside the
/// closure but the writer still receives the buffer `inout`.
@inline(never)
func lostStoreCanaryObservedInsideBorrow() -> Float {
    var values: [Float] = [2]
    return values.withUnsafeMutableBufferPointer { buffer in
        lostStoreCanaryWriteOneInout(&buffer)
        return buffer[0]
    }
}

/// Control: the by-value writer, the only kind VectorAccelerate's own borrows call.
@inline(never)
func lostStoreControlByValueWriter() -> Float {
    var values: [Float] = [2]
    values.withUnsafeMutableBufferPointer { lostStoreCanaryWriteOneByValue($0) }
    return values[0]
}

/// Control: a scalar out-parameter handed by value to an opaque callee and read inside the borrow, the
/// `PerformanceMonitor.updateResourceMetrics` idiom for `task_info`.
@inline(never)
func lostStoreControlOutPointerReadInside() -> Float {
    var value: Float = 2
    return withUnsafeMutablePointer(to: &value) { pointer -> Float in
        lostStoreCanaryWriteOneThroughPointer(pointer)
        return pointer.pointee
    }
}

/// Control: an accumulator captured and written inside a read-only borrow, then read after it. This is the
/// shape of the CPU fallbacks' sums (`SIMDFallback`, `SIMDOptimizer`, `StableNormalization.normalizedSIMD`,
/// `CPUTopK`), the form VectorCore's guard flags by class and never reproduced.
@inline(never)
func lostStoreControlCapturedAccumulator() -> Float {
    let values: [Float] = [0.5, 0.25, 0.125, 0.125]
    var sum: Float = 0
    values.withUnsafeBufferPointer { buffer in
        for x in buffer { sum += x }
    }
    return sum
}

// MARK: - Tests

final class CompilerLostStoreCanaryTests: XCTestCase {

    /// The compilers on which VectorCore's investigation demonstrated the miscompile (6.2 and 6.4.2).
    private static let compilerInDemonstratedRange: Bool = {
        #if compiler(>=6.2) && !compiler(>=6.5)
        return true
        #else
        return false
        #endif
    }()

    private static let isOptimizedBuild: Bool = {
        #if DEBUG
        return false
        #else
        return true
        #endif
    }()

    private static func detail(_ observed: Float) -> String {
        "observed \(observed) (bits 0x\(String(observed.bitPattern, radix: 16))), expected 1 (bits 0x3f800000)"
    }

    /// Asserts the store was observed. In an optimized build on a demonstrated compiler a lost store is an
    /// expected, non-strict failure: recorded in the log, not a red gate.
    private func assertStoreObserved(_ observed: Float, _ shape: String,
                                     file: StaticString = #filePath, line: UInt = #line) {
        let message = "VC-C001 \(shape): \(Self.detail(observed))"
        if Self.isOptimizedBuild && Self.compilerInDemonstratedRange {
            let options = XCTExpectedFailure.Options()
            options.isStrict = false
            XCTExpectFailure("VC-C001: Swift optimizer lost store (RedundantLoadElimination) on a compiler where "
                             + "VectorCore demonstrated it; \(shape)", options: options) {
                XCTAssertEqual(observed.bitPattern, Float(1).bitPattern, message, file: file, line: line)
            }
        } else {
            XCTAssertEqual(observed.bitPattern, Float(1).bitPattern, message, file: file, line: line)
        }
    }

    func testLiteralArrayStoreIsObservedAfterOutOfLineInoutWriter() {
        assertStoreObserved(lostStoreCanaryObservedAfterBorrow(), "post-borrow read")
    }

    func testLiteralArrayStoreIsObservedWhenReadInsideBorrowAfterInoutWriter() {
        assertStoreObserved(lostStoreCanaryObservedInsideBorrow(), "inside-borrow read")
    }

    // Controls: must hold in every build on every toolchain.

    func testByValueWriterControlKeepsTheStore() {
        XCTAssertEqual(lostStoreControlByValueWriter().bitPattern, Float(1).bitPattern,
                       "by-value writer: \(Self.detail(lostStoreControlByValueWriter()))")
    }

    func testOutPointerReadInsideControlKeepsTheStore() {
        XCTAssertEqual(lostStoreControlOutPointerReadInside().bitPattern, Float(1).bitPattern,
                       "out-pointer read inside the borrow: \(Self.detail(lostStoreControlOutPointerReadInside()))")
    }

    func testCapturedAccumulatorControlKeepsTheWrites() {
        XCTAssertEqual(lostStoreControlCapturedAccumulator().bitPattern, Float(1).bitPattern,
                       "captured accumulator: \(Self.detail(lostStoreControlCapturedAccumulator()))")
    }
}
