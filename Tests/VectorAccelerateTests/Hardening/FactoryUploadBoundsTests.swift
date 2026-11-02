import XCTest
import Metal
import VectorCore
@testable import VectorAccelerate

/// A vector backed by a larger allocation whose suffix is outside its exposed payload.
/// Copying allocation padding from the source leaks the controlled suffix into the GPU buffer.
private struct UploadWindowVector: VectorProtocol {
    typealias Scalar = Float
    var storage: [Float]
    var scalarCount: Int
    var count: Int { scalarCount }
    // Swift 6.3.3's release SIL linker crashes on the inherited VectorCore getter
    // for this new conformer. An explicit equivalent subscript avoids that compiler bug.
    subscript(index: Int) -> Float {
        precondition(index >= 0 && index < scalarCount)
        return storage[index]
    }

    init() { storage = []; scalarCount = 0 }
    init(_ array: [Float]) { storage = array; scalarCount = array.count }
    init(repeating value: Float) { storage = [value]; scalarCount = 1 }
    init(payload: [Float], suffix: [Float], declaredCount: Int? = nil) {
        storage = payload + suffix
        scalarCount = declaredCount ?? payload.count
    }
    func toArray() -> [Float] { Array(storage.prefix(Swift.max(0, Swift.min(scalarCount, storage.count)))) }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer { ptr in
            try body(UnsafeBufferPointer(start: ptr.baseAddress, count: Swift.max(0, Swift.min(scalarCount, ptr.count))))
        }
    }
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        let limit = scalarCount
        return try storage.withUnsafeMutableBufferPointer { ptr in
            try body(UnsafeMutableBufferPointer(start: ptr.baseAddress, count: Swift.max(0, Swift.min(limit, ptr.count))))
        }
    }
}

final class FactoryUploadBoundsTests: XCTestCase {
    private func factory() throws -> MetalBufferFactory {
        MetalBufferFactory(device: try XCTUnwrap(MTLCreateSystemDefaultDevice()))
    }

    private func floats(_ buffer: any MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }

    func testInvalidAlignmentIsRejected() throws {
        let factory = try factory()
        XCTAssertNil(factory.createAlignedBuffer(length: 7, alignment: 3))
        XCTAssertNil(factory.createAlignedBuffer(from: [Float(1), 2, 3], alignment: 3))
        XCTAssertNil(factory.createBuffer(fromVector: UploadWindowVector([1, 2, 3]), alignment: 3))
        XCTAssertNil(factory.createBuffer(fromVectors: [UploadWindowVector([1, 2, 3])], alignment: 3))
    }

    func testUnrepresentableAlignedLengthIsRejectedWithoutOverflow() throws {
        let factory = try factory()
        XCTAssertNil(factory.createAlignedBuffer(length: Int.max, alignment: 16))
    }

    func testInvalidSizesAndAlignmentDoNotReachAllocationOrCopy() throws {
        let factory = try factory()
        let vector = UploadWindowVector([1, 2, 3])
        for alignment in [0, -1, 3, Int.max] {
            XCTAssertNil(factory.createAlignedBuffer(length: 12, alignment: alignment))
            XCTAssertNil(factory.createAlignedBuffer(from: [Float(1), 2, 3], alignment: alignment))
            XCTAssertNil(factory.createBuffer(fromVector: vector, alignment: alignment))
            XCTAssertNil(factory.createBuffer(fromVectors: [vector], alignment: alignment))
        }
        for length in [0, -1, Int.min, Int.max, factory.device.maxBufferLength + 1] {
            XCTAssertNil(factory.createAlignedBuffer(length: length))
        }
        for count in [-1, Int.max] {
            let invalid = UploadWindowVector(payload: [1], suffix: [], declaredCount: count)
            XCTAssertNil(factory.createBuffer(fromVector: invalid))
            XCTAssertNil(factory.createBuffer(fromVectors: [invalid, invalid]))
        }
        XCTAssertNil(factory.createAlignedBuffer(from: [Float]()))
        XCTAssertNil(factory.createBuffer(fromVector: UploadWindowVector()))
        XCTAssertNil(factory.createBuffer(fromVectors: [UploadWindowVector]()))
        XCTAssertNil(factory.createBuffer(fromVectors: [UploadWindowVector()]))
    }

    func testDeclaredCountMustMatchExposedSourceStorage() throws {
        let factory = try factory()
        let short = UploadWindowVector(payload: [1, 2, 3], suffix: [], declaredCount: 4)
        XCTAssertNil(factory.createBuffer(fromVector: short))
        XCTAssertNil(factory.createBuffer(fromVectors: [UploadWindowVector([1, 2, 3, 4]), short]))
    }

    func testInitializedUploadsRejectCPUInaccessibleStorage() throws {
        let factory = try factory()
        let vector = UploadWindowVector([1, 2, 3])
        for options: MTLResourceOptions in [.storageModePrivate, .storageModeMemoryless] {
            XCTAssertNil(factory.createAlignedBuffer(from: [Float(1)], options: options))
            XCTAssertNil(factory.createBuffer(fromVector: vector, options: options))
            XCTAssertNil(factory.createBuffer(fromVectors: [vector], options: options))
            XCTAssertNil(factory.createBufferPair([vector], [vector], options: options))
        }
        // Uninitialized private allocation remains supported; it does not need a CPU copy.
        let privateBuffer = try XCTUnwrap(factory.createAlignedBuffer(length: 12, options: .storageModePrivate))
        XCTAssertEqual(privateBuffer.length, 16)
        XCTAssertEqual(privateBuffer.storageMode, .private)
    }

    func testBytePayloadAndResourceOptionsArePreserved() throws {
        let factory = try factory()
        let options: MTLResourceOptions = [.storageModeShared, .cpuCacheModeWriteCombined, .hazardTrackingModeUntracked]
        let result = try XCTUnwrap(factory.createAlignedBuffer(from: [UInt8(0x12), 0x34, 0x56], alignment: 8, options: options))
        XCTAssertEqual(result.length, 8)
        XCTAssertEqual(result.storageMode, .shared)
        XCTAssertEqual(result.cpuCacheMode, .writeCombined)
        XCTAssertEqual(result.hazardTrackingMode, .untracked)
        let bytes = UnsafeBufferPointer(start: result.contents().assumingMemoryBound(to: UInt8.self), count: 8)
        XCTAssertEqual(Array(bytes), [0x12, 0x34, 0x56, 0, 0, 0, 0, 0])
    }

    func testUploadedBuffersRemainValidForGPUReadsAfterSourceLifetimeEnds() throws {
        let factory = try factory()
        func upload() throws -> any MTLBuffer {
            let vector = UploadWindowVector(payload: [1, -2, 3], suffix: [12345])
            return try XCTUnwrap(factory.createBuffer(fromVector: vector, alignment: 16))
        }
        let source = try upload()
        let destination = try XCTUnwrap(factory.device.makeBuffer(length: 16, options: .storageModeShared))
        destination.contents().initializeMemory(as: UInt8.self, repeating: 0xCD, count: 16)
        let queue = try XCTUnwrap(factory.device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
        blit.copy(from: source, sourceOffset: 0, to: destination, destinationOffset: 0, size: 16)
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
        XCTAssertEqual(floats(destination, count: 4), [1, -2, 3, 0])
        let pair = try XCTUnwrap(factory.createBufferPair([UploadWindowVector([1, 2, 3])], [UploadWindowVector([4, 5])]))
        XCTAssertEqual(floats(pair.bufferA, count: 4), [1, 2, 3, 0])
        XCTAssertEqual(floats(pair.bufferB, count: 4), [4, 5, 0, 0])
    }

    func testVectorUploadDoesNotCopyOutsideExposedPayload() throws {
        let factory = try factory()
        let vector = UploadWindowVector(payload: [1, -2, 3], suffix: [12345])
        let result = try XCTUnwrap(factory.createBuffer(fromVector: vector, alignment: 16))
        XCTAssertEqual(result.length, 16)
        XCTAssertEqual(floats(result, count: 4), [1, -2, 3, 0])
        XCTAssertEqual(vector.storage, [1, -2, 3, 12345])
    }

    func testAlignedArrayCopiesPayloadAndZeroesPadding() throws {
        let factory = try factory()
        // Removing a trivial element leaves reserved storage outside the array's count.
        // It must never become uploaded payload or destination padding.
        var shortened: [Float] = [1, -2, 3, 12345]
        shortened.removeLast()
        let shortenedResult = try XCTUnwrap(factory.createAlignedBuffer(from: shortened, alignment: 16))
        XCTAssertEqual(floats(shortenedResult, count: 4), [1, -2, 3, 0])
        for (payload, alignment, length) in [([Float(7)], 16, 16), ([1, -2, 3], 16, 16), ([1, 2, 3, 4], 16, 16), ([1, 2, 3, 4, 5], 32, 32)] {
            let result = try XCTUnwrap(factory.createAlignedBuffer(from: payload, alignment: alignment))
            XCTAssertEqual(result.length, length)
            XCTAssertEqual(floats(result, count: payload.count), payload)
            let padding = result.contents().advanced(by: payload.count * 4).assumingMemoryBound(to: UInt8.self)
            XCTAssertTrue(UnsafeBufferPointer(start: padding, count: length - payload.count * 4).allSatisfy { $0 == 0 })
        }
    }

    func testVectorBatchesRejectRaggedRowsInsteadOfTruncatingOrLeavingHoles() throws {
        let factory = try factory()
        let row = UploadWindowVector([1, 2, 3])
        for other in [UploadWindowVector([4, 5]), UploadWindowVector([4, 5, 6, 7])] {
            XCTAssertNil(factory.createBuffer(fromVectors: [row, other]))
            XCTAssertNil(factory.createBufferPair([row], [row, other]))
        }
        let result = try XCTUnwrap(factory.createBuffer(fromVectors: [row, UploadWindowVector([4, 5, 6])]))
        XCTAssertEqual(result.length, 32)
        XCTAssertEqual(floats(result, count: 8), [1, 2, 3, 4, 5, 6, 0, 0])
    }
}
