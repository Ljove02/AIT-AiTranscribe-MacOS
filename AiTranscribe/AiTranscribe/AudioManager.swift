/*
 AudioManager.swift
 ==================

 Manages audio output ducking/muting during recording.

 Uses AppleScript commands for system volume control.
 No additional permissions required.

 Thread-safe with MainActor isolation for UI updates.
 */

import Foundation
import Combine

/// Manages audio ducking for clean recordings during recording sessions
@MainActor
final class AudioManager: ObservableObject {
    
    // MARK: - Singleton
    
    static let shared = AudioManager()
    
    // MARK: - Published Properties (for UI only)
    
    /// Current ducking state - safe to observe from SwiftUI
    @Published private(set) var isDucking: Bool = false
    
    // MARK: - Private Properties
    
    /// Original volume before ducking - stored for restoration
    private var originalVolume: Float = 1.0
    
    /// Whether we're currently in a ducking session
    private var isInRecordingSession: Bool = false
    
    /// The target volume percentage for current session
    private var sessionTargetVolume: Double = 0.0
    
    /// Lock object for thread safety
    private let lock = NSLock()
    
    // MARK: - Initialization
    
    private init() {}
    
    deinit {
        // Note: deinit runs in nonisolated context, so we do direct cleanup
        // This is safe because deinit only runs when the object is being destroyed
        // and no other code can access it
        if isInRecordingSession {
            let targetVolume = Int(originalVolume * 100)
            let script = "set volume output volume \(targetVolume)"
            
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
            task.arguments = ["-e", script]
            task.standardOutput = Pipe()
            task.standardError = Pipe()
            
            try? task.run()
            task.waitUntilExit()
        }
    }
    
    // MARK: - Public Methods - Session-Based API
    
    /// Start a ducking session (call when recording starts)
    /// - Parameter reductionPercentage: How much to lower volume (0-99)
    ///   Example: If reduction is 80%, volume becomes original × 20%
    func startDuckingSession(reductionPercentage: Double) {
        lock.lock()
        
        // Prevent multiple duck calls during one session
        guard !isInRecordingSession else {
            lock.unlock()
            print("AudioManager: Already in ducking session, ignoring duplicate call")
            return
        }
        
        // Store original volume
        originalVolume = getSystemVolume()
        isInRecordingSession = true
        
        // Calculate target volume: reduced by the percentage
        // Example: 80% reduction on 100% = 20%, 80% reduction on 50% = 10%
        let reductionFactor = max(0.01, min(1.0, reductionPercentage / 100.0))
        var targetVolume = Double(originalVolume) * (1.0 - reductionFactor)
        
        // Smart fallback: Never boost volume
        // If calculation would boost (shouldn't happen with this formula, but be safe)
        if targetVolume > Double(originalVolume) {
            targetVolume = Double(originalVolume) * 0.5  // Safe fallback: reduce by 50%
            print("AudioManager: Smart fallback triggered, reducing to 50% instead")
        }
        
        // Convert from 0-1 to 0-100 for AppleScript
        let targetPercentage = targetVolume * 100.0
        
        lock.unlock()
        
        // Apply ducking (outside lock to avoid blocking)
        applyVolumeChange(to: targetPercentage)
        
        // Update published property asynchronously to prevent "Publishing changes from within view updates"
        // Use DispatchQueue.main.async to truly defer to next run loop (Task doesn't defer on MainActor)
        DispatchQueue.main.async { [weak self] in
            self?.isDucking = true
        }
        
        print("AudioManager: Started ducking session - reduced by \(Int(reductionPercentage))% (original: \(Int(originalVolume * 100))%, target: \(Int(targetPercentage))%)")
    }
    
    /// End a ducking session (call when recording stops)
    func endDuckingSession() {
        lock.lock()
        
        guard isInRecordingSession else {
            lock.unlock()
            print("AudioManager: Not in ducking session, nothing to restore")
            return
        }
        
        let volumeToRestore = originalVolume
        isInRecordingSession = false
        
        lock.unlock()
        
        // Restore original volume (outside lock)
        restoreToOriginalVolume()
        
        // Update published property asynchronously to prevent "Publishing changes from within view updates"
        // Use DispatchQueue.main.async to truly defer to next run loop (Task doesn't defer on MainActor)
        DispatchQueue.main.async { [weak self] in
            self?.isDucking = false
        }
        
        print("AudioManager: Ended ducking session, restored to \(Int(volumeToRestore * 100))%")
    }
    
    /// Check if we're currently ducking (safe for UI)
    var currentlyDucking: Bool {
        lock.lock()
        defer { lock.unlock() }
        return isInRecordingSession && isDucking
    }
    
    /// Get the original volume before ducking
    var volumeBeforeDucking: Float {
        lock.lock()
        defer { lock.unlock() }
        return originalVolume
    }
    
    // MARK: - Public Methods - Direct API (for advanced use)
    
    /// Duck (lower) system audio output directly
    /// - Parameter percentage: 0.0 = mute, 100.0 = full volume
    func duckAudio(to percentage: Double) {
        lock.lock()
        
        originalVolume = getSystemVolume()
        isInRecordingSession = false
        sessionTargetVolume = percentage
        
        let originalVol = originalVolume
        
        lock.unlock()
        
        applyVolumeChange(to: percentage)
        
        // Update published property asynchronously to prevent "Publishing changes from within view updates"
        // Use DispatchQueue.main.async to truly defer to next run loop (Task doesn't defer on MainActor)
        DispatchQueue.main.async { [weak self] in
            self?.isDucking = true
        }
        
        print("AudioManager: Ducked to \(Int(percentage))% (was \(Int(originalVol * 100))%)")
    }
    
    /// Mute system audio output directly
    func muteAudio() {
        duckAudio(to: 0.0)
    }
    
    /// Restore original audio volume directly
    func restoreAudio() {
        lock.lock()
        
        let volumeToRestore = originalVolume
        isInRecordingSession = false
        
        lock.unlock()
        
        restoreToOriginalVolume()
        
        // Update published property asynchronously to prevent "Publishing changes from within view updates"
        // Use DispatchQueue.main.async to truly defer to next run loop (Task doesn't defer on MainActor)
        DispatchQueue.main.async { [weak self] in
            self?.isDucking = false
        }
        
        print("AudioManager: Restored to \(Int(volumeToRestore * 100))%")
    }
    
    // MARK: - Private Methods
    
    /// Get current system output volume (0.0 to 1.0)
    private func getSystemVolume() -> Float {
        let task = Process()
        let outputPipe = Pipe()
        
        task.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        task.arguments = ["-e", "output volume of (get volume settings)"]
        task.standardOutput = outputPipe
        task.standardError = Pipe()
        
        do {
            try task.run()
            task.waitUntilExit()
            
            guard task.terminationStatus == 0 else {
                return 1.0
            }
            
            let data = outputPipe.fileHandleForReading.readDataToEndOfFile()
            guard let output = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                  let vol = Float(output) else {
                return 1.0
            }
            
            // Clamp to valid range
            return max(0.0, min(100.0, vol)) / 100.0
            
        } catch {
            print("AudioManager: Failed to get volume: \(error)")
            return 1.0
        }
    }
    
    /// Apply volume change using AppleScript
    /// - Parameter percentage: Target volume percentage (0.0 to 100.0)
    private func applyVolumeChange(to percentage: Double) {
        let clampedPercentage = max(0.0, min(100.0, percentage))
        let volumeInt = Int(clampedPercentage)
        
        // AppleScript to set output volume
        let script = "set volume output volume \(volumeInt)"
        
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        task.arguments = ["-e", script]
        task.standardOutput = Pipe()
        task.standardError = Pipe()
        
        do {
            try task.run()
            task.waitUntilExit()
            
            if task.terminationStatus != 0 {
                print("AudioManager: Failed to set volume to \(volumeInt)%")
            }
        } catch {
            print("AudioManager: Error setting volume: \(error)")
        }
    }
    
    /// Restore to the original volume that was stored before ducking
    private func restoreToOriginalVolume() {
        let targetVolume = Int(originalVolume * 100)
        
        let script = "set volume output volume \(targetVolume)"
        
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        task.arguments = ["-e", script]
        task.standardOutput = Pipe()
        task.standardError = Pipe()
        
        do {
            try task.run()
            task.waitUntilExit()
            
            if task.terminationStatus != 0 {
                print("AudioManager: Failed to restore volume to \(targetVolume)%")
            }
        } catch {
            print("AudioManager: Error restoring volume: \(error)")
        }
    }
}
