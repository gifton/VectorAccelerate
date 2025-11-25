// VectorAccelerate: Logging System
//
// Comprehensive logging for debugging and performance monitoring

import Foundation
import os.log

/// Log levels for VectorAccelerate
public enum LogLevel: Int, Comparable, Sendable {
    case verbose = 0
    case debug = 1
    case info = 2
    case warning = 3
    case error = 4
    case critical = 5
    
    public static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
    
    var symbol: String {
        switch self {
        case .verbose: return "🔍"
        case .debug: return "🐛"
        case .info: return "ℹ️"
        case .warning: return "⚠️"
        case .error: return "❌"
        case .critical: return "🔥"
        }
    }
    
    var osLogType: OSLogType {
        switch self {
        case .verbose, .debug: return .debug
        case .info: return .info
        case .warning: return .default
        case .error: return .error
        case .critical: return .fault
        }
    }
}

/// Logger configuration
public struct LoggerConfiguration: Sendable {
    public let minimumLevel: LogLevel
    public let enableConsole: Bool
    public let enableFile: Bool
    public let enableMetrics: Bool
    public let fileURL: URL?
    
    public init(
        minimumLevel: LogLevel = .info,
        enableConsole: Bool = true,
        enableFile: Bool = false,
        enableMetrics: Bool = true,
        fileURL: URL? = nil
    ) {
        self.minimumLevel = minimumLevel
        self.enableConsole = enableConsole
        self.enableFile = enableFile
        self.enableMetrics = enableMetrics
        self.fileURL = fileURL
    }
    
    public static let `default` = LoggerConfiguration()
    public static let debug = LoggerConfiguration(minimumLevel: .debug)
    public static let production = LoggerConfiguration(minimumLevel: .warning, enableMetrics: true)
}

/// Main logger for VectorAccelerate
public actor Logger {
    private let configuration: LoggerConfiguration
    private let osLog: OSLog
    private var metrics: [String: Int] = [:]
    private var performanceMetrics: [String: [TimeInterval]] = [:]
    private let dateFormatter: DateFormatter
    
    // Singleton instance
    public static let shared = Logger()
    
    // Global configuration
    private nonisolated(unsafe) static var globalConfig = LoggerConfiguration.default
    
    public init(configuration: LoggerConfiguration = LoggerConfiguration.default) {
        self.configuration = configuration
        self.osLog = OSLog(subsystem: "com.vectoraccelerate", category: "VectorAccelerate")
        
        self.dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        
        // Set global config
        Logger.globalConfig = configuration
    }
    
    // MARK: - Configuration
    
    public static func configure(_ configuration: LoggerConfiguration) {
        globalConfig = configuration
    }
    
    // MARK: - Logging Methods
    
    public func log(
        _ message: String,
        level: LogLevel = .info,
        category: String? = nil,
        file: String = #file,
        function: String = #function,
        line: Int = #line
    ) {
        guard level >= configuration.minimumLevel else { return }
        
        let fileName = URL(fileURLWithPath: file).lastPathComponent
        let location = "\(fileName):\(line)"
        let categoryStr = category.map { "[\($0)] " } ?? ""
        
        let formattedMessage = "\(level.symbol) \(categoryStr)\(message) - \(function) @ \(location)"
        
        // Console logging
        if configuration.enableConsole {
            os_log("%{public}@", log: osLog, type: level.osLogType, formattedMessage)
        }
        
        // File logging
        if configuration.enableFile, let fileURL = configuration.fileURL {
            let timestamp = dateFormatter.string(from: Date())
            let logEntry = "\(timestamp) \(formattedMessage)\n"
            
            Task.detached {
                do {
                    let fileHandle = try FileHandle(forWritingTo: fileURL)
                    fileHandle.seekToEndOfFile()
                    if let data = logEntry.data(using: .utf8) {
                        fileHandle.write(data)
                    }
                    fileHandle.closeFile()
                } catch {
                    // Create file if it doesn't exist
                    try? logEntry.write(to: fileURL, atomically: true, encoding: .utf8)
                }
            }
        }
        
        // Track metrics
        if configuration.enableMetrics {
            let key = "\(level)_count"
            metrics[key, default: 0] += 1
        }
    }
    
    // MARK: - Convenience Methods
    
    public func verbose(_ message: String, category: String? = nil, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, level: .verbose, category: category, file: file, function: function, line: line)
    }
    
    public func debug(_ message: String, category: String? = nil, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, level: .debug, category: category, file: file, function: function, line: line)
    }
    
    public func info(_ message: String, category: String? = nil, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, level: .info, category: category, file: file, function: function, line: line)
    }
    
    public func warning(_ message: String, category: String? = nil, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, level: .warning, category: category, file: file, function: function, line: line)
    }
    
    public func error(_ message: String, category: String? = nil, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, level: .error, category: category, file: file, function: function, line: line)
    }
    
    public func critical(_ message: String, category: String? = nil, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, level: .critical, category: category, file: file, function: function, line: line)
    }
    
    // MARK: - Performance Logging
    
    public func logPerformance(
        _ operation: String,
        duration: TimeInterval,
        metadata: [String: String]? = nil
    ) {
        performanceMetrics[operation, default: []].append(duration)
        
        let metadataStr = metadata.map { dict in
            dict.map { "\($0.key)=\($0.value)" }.joined(separator: ", ")
        } ?? ""
        
        info("Performance: \(operation) took \(String(format: "%.3f", duration * 1000))ms \(metadataStr)", category: "Performance")
    }
    
    public func startMeasure(_ operation: String) -> MeasureToken {
        MeasureToken(operation: operation, logger: self)
    }
    
    // MARK: - Metrics
    
    public func getMetrics() -> [String: Int] {
        return metrics
    }
    
    public func getPerformanceStats(for operation: String) -> LoggerPerformanceStats? {
        guard let durations = performanceMetrics[operation], !durations.isEmpty else {
            return nil
        }
        
        let sorted = durations.sorted()
        let sum = durations.reduce(0, +)
        let average = sum / Double(durations.count)
        let median = sorted[sorted.count / 2]
        let p95 = sorted[Int(Double(sorted.count) * 0.95)]
        let min = sorted.first!
        let max = sorted.last!
        
        return LoggerPerformanceStats(
            count: durations.count,
            average: average,
            median: median,
            p95: p95,
            min: min,
            max: max
        )
    }
}

/// Token for measuring performance
public final class MeasureToken: @unchecked Sendable {
    private let operation: String
    private let startTime: Date
    private weak var logger: Logger?
    private var metadata: [String: String] = [:]
    private let lock = NSLock()
    
    init(operation: String, logger: Logger) {
        self.operation = operation
        self.startTime = Date()
        self.logger = logger
    }
    
    public func addMetadata(_ key: String, value: String) {
        lock.lock()
        defer { lock.unlock() }
        metadata[key] = value
    }
    
    public func end() {
        let duration = Date().timeIntervalSince(startTime)
        lock.lock()
        let capturedMetadata = metadata
        let capturedOperation = operation
        let capturedLogger = logger
        lock.unlock()
        
        Task.detached {
            await capturedLogger?.logPerformance(capturedOperation, duration: duration, metadata: capturedMetadata)
        }
    }
    
    deinit {
        // Auto-end if not explicitly ended
        end()
    }
}

/// Logger performance statistics
public struct LoggerPerformanceStats: Sendable {
    public let count: Int
    public let average: TimeInterval
    public let median: TimeInterval
    public let p95: TimeInterval
    public let min: TimeInterval
    public let max: TimeInterval
}

// MARK: - Global Logging Functions

public func VectorLog(
    _ message: String,
    level: LogLevel = .info,
    category: String? = nil,
    file: String = #file,
    function: String = #function,
    line: Int = #line
) {
    Task {
        await Logger.shared.log(message, level: level, category: category, file: file, function: function, line: line)
    }
}

public func VectorLogDebug(
    _ message: String,
    category: String? = nil,
    file: String = #file,
    function: String = #function,
    line: Int = #line
) {
    Task {
        await Logger.shared.debug(message, category: category, file: file, function: function, line: line)
    }
}

public func VectorLogError(
    _ message: String,
    category: String? = nil,
    file: String = #file,
    function: String = #function,
    line: Int = #line
) {
    Task {
        await Logger.shared.error(message, category: category, file: file, function: function, line: line)
    }
}