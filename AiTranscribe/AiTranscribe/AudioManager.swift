/*
 AudioManager.swift
 ==================

 Manages audio output ducking/muting during recording.

 Uses CoreAudio for system volume control (no process spawns on the hot path),
 with an AppleScript fallback for devices that don't expose a settable output volume.
 No additional permissions required.

 Thread-safe with MainActor isolation for UI updates.
 */

import Foundation
import Combine
import CoreAudio

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

    /// Output device the current session ducked
    private var duckedDeviceID: AudioDeviceID?

    /// Volume multiplier applied while ducking (e.g. 0.3 for a 70% reduction)
    private var duckMultiplier: Float = 1.0

    /// Original volume per output profile, keyed by the device's nominal sample rate.
    /// Bluetooth devices with a microphone (HFP) flip from the A2DP music profile
    /// (44.1/48 kHz) to the call profile (16/24 kHz) when a mic opens, and each
    /// profile keeps its own volume. Restoring one value into the wrong profile
    /// is what left music stuck at the ducked level after recording.
    private var originalVolumeByProfile: [Float64: Float] = [:]

    /// Removes the sample-rate listener on the ducked device
    private var removeProfileListener: (() -> Void)?

    /// Ends the post-recording restore window
    private var restoreWindowWork: DispatchWorkItem?

    /// How long to keep re-applying the original volume after recording stops.
    /// Bluetooth takes ~3s to switch back from HFP to A2DP after the mic closes.
    private let restoreWindow: TimeInterval = 8.0

    /// Lock object for thread safety
    private let lock = NSLock()

    // MARK: - Initialization

    private init() {}

    deinit {
        // Note: deinit runs in nonisolated context, so we do direct cleanup
        // This is safe because deinit only runs when the object is being destroyed
        // and no other code can access it
        if isInRecordingSession {
            _ = AudioManager.writeCoreAudioVolume(originalVolume)
        }
    }

    // MARK: - Public Methods - Session-Based API

    /// Start a ducking session (call when recording starts)
    /// - Parameter reductionPercentage: How much to lower volume (0-99)
    ///   Example: If reduction is 80%, volume becomes original × 20%
    func startDuckingSession(reductionPercentage: Double) {
        // Prevent multiple duck calls during one session
        guard !isInRecordingSession else {
            print("AudioManager: Already in ducking session, ignoring duplicate call")
            return
        }

        let reductionFactor = max(0.01, min(1.0, reductionPercentage / 100.0))
        duckMultiplier = Float(1.0 - reductionFactor)
        isInRecordingSession = true

        guard let deviceID = Self.defaultOutputDeviceID(),
              let currentVolume = Self.readCoreAudioVolume(deviceID: deviceID) else {
            // No CoreAudio volume control — single-value AppleScript path
            endRestoreWindow()
            originalVolume = getSystemVolumeViaOSAScript()
            applyVolumeChangeViaOSAScript(to: Double(originalVolume * duckMultiplier) * 100.0)
            setDucking(true)
            print("AudioManager: Started ducking session via AppleScript - reduced by \(Int(reductionPercentage))% (original: \(Int(originalVolume * 100))%)")
            return
        }

        let profile = Self.nominalSampleRate(deviceID: deviceID)

        if duckedDeviceID == deviceID, restoreWindowWork != nil {
            // Recording again while the last restore is still settling (the
            // device may still be in its call profile). Keep the originals we
            // already know rather than capturing a ducked value as "original".
            restoreWindowWork?.cancel()
            restoreWindowWork = nil
        } else {
            endRestoreWindow()
            duckedDeviceID = deviceID
            originalVolumeByProfile = [:]
            installProfileListener(deviceID: deviceID)
        }

        if originalVolumeByProfile[profile] == nil {
            originalVolumeByProfile[profile] = currentVolume
        }
        let original = originalVolumeByProfile[profile] ?? currentVolume
        originalVolume = original
        let target = original * duckMultiplier
        _ = Self.writeCoreAudioVolume(target, deviceID: deviceID)
        setDucking(true)

        print("AudioManager: Started ducking session - reduced by \(Int(reductionPercentage))% (original: \(Int(original * 100))%, target: \(Int(target * 100))%, profile: \(Int(profile)) Hz)")
    }

    /// End a ducking session (call when recording stops)
    func endDuckingSession() {
        guard isInRecordingSession else {
            print("AudioManager: Not in ducking session, nothing to restore")
            return
        }
        isInRecordingSession = false
        setDucking(false)

        guard let deviceID = duckedDeviceID else {
            restoreToOriginalVolumeViaOSAScript()
            print("AudioManager: Ended ducking session, restored to \(Int(originalVolume * 100))%")
            return
        }

        restoreCurrentProfile(deviceID: deviceID)

        // Keep the profile listener alive briefly: when the Bluetooth device
        // drops back from HFP to A2DP, the listener restores that profile too.
        let work = DispatchWorkItem { [weak self] in
            MainActor.assumeIsolated { self?.endRestoreWindow() }
        }
        restoreWindowWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + restoreWindow, execute: work)

        print("AudioManager: Ended ducking session, restoring originals \(originalVolumeByProfile.map { "\(Int($0.key)) Hz: \(Int($0.value * 100))%" })")
    }

    private func setDucking(_ value: Bool) {
        // Update published property asynchronously to prevent "Publishing changes from within view updates"
        // Use DispatchQueue.main.async to truly defer to next run loop (Task doesn't defer on MainActor)
        DispatchQueue.main.async { [weak self] in
            self?.isDucking = value
        }
    }

    /// Write back the original volume for whichever profile the device is in now
    private func restoreCurrentProfile(deviceID: AudioDeviceID) {
        let profile = Self.nominalSampleRate(deviceID: deviceID)
        guard let original = originalVolumeByProfile[profile] else { return }
        if !Self.writeCoreAudioVolume(original, deviceID: deviceID) {
            print("AudioManager: Failed to restore \(Int(profile)) Hz profile to \(Int(original * 100))%")
        }
    }

    /// The ducked device switched profiles (sample rate changed)
    private func handleProfileChange() {
        guard let deviceID = duckedDeviceID else { return }
        let profile = Self.nominalSampleRate(deviceID: deviceID)

        if isInRecordingSession {
            // New profile mid-recording (A2DP -> HFP): remember its own
            // original volume, then duck it as well.
            if originalVolumeByProfile[profile] == nil,
               let current = Self.readCoreAudioVolume(deviceID: deviceID) {
                originalVolumeByProfile[profile] = current
            }
            if let original = originalVolumeByProfile[profile] {
                _ = Self.writeCoreAudioVolume(original * duckMultiplier, deviceID: deviceID)
                print("AudioManager: Output switched to \(Int(profile)) Hz profile, ducked \(Int(original * 100))% -> \(Int(original * duckMultiplier * 100))%")
            }
        } else if restoreWindowWork != nil {
            // Back to a profile we ducked earlier (HFP -> A2DP): restore it
            restoreCurrentProfile(deviceID: deviceID)
            print("AudioManager: Output switched to \(Int(profile)) Hz profile, restored")
        }
    }

    private func installProfileListener(deviceID: AudioDeviceID) {
        removeProfileListener?()
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyNominalSampleRate,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            // Let the HAL finish the profile switch before touching volume
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                MainActor.assumeIsolated { self?.handleProfileChange() }
            }
        }
        guard AudioObjectAddPropertyListenerBlock(deviceID, &address, DispatchQueue.main, block) == noErr else {
            return
        }
        removeProfileListener = {
            var address = address
            AudioObjectRemovePropertyListenerBlock(deviceID, &address, DispatchQueue.main, block)
        }
    }

    private func endRestoreWindow() {
        restoreWindowWork?.cancel()
        restoreWindowWork = nil
        removeProfileListener?()
        removeProfileListener = nil
        duckedDeviceID = nil
        originalVolumeByProfile = [:]
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
        if let volume = Self.readCoreAudioVolume() {
            return volume
        }
        return getSystemVolumeViaOSAScript()
    }

    /// Apply volume change
    /// - Parameter percentage: Target volume percentage (0.0 to 100.0)
    private func applyVolumeChange(to percentage: Double) {
        let clampedPercentage = max(0.0, min(100.0, percentage))

        if Self.writeCoreAudioVolume(Float(clampedPercentage / 100.0)) {
            return
        }
        applyVolumeChangeViaOSAScript(to: clampedPercentage)
    }

    /// Restore to the original volume that was stored before ducking
    private func restoreToOriginalVolume() {
        if Self.writeCoreAudioVolume(originalVolume) {
            return
        }
        restoreToOriginalVolumeViaOSAScript()
    }

    // MARK: - Private Methods - AppleScript Fallback

    /// Get current system output volume via osascript (0.0 to 1.0)
    private func getSystemVolumeViaOSAScript() -> Float {
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
    private func applyVolumeChangeViaOSAScript(to percentage: Double) {
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
    
    /// Restore to the original volume via osascript
    private func restoreToOriginalVolumeViaOSAScript() {
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

    // MARK: - Private Methods - CoreAudio Fast Path

    /// Same FourCC as kAudioHardwareServiceDeviceProperty_VirtualMainVolume ('vmvc'),
    /// defined locally so we only use the non-deprecated AudioObject APIs
    private nonisolated static let virtualMainVolumeSelector = AudioObjectPropertySelector(0x766D_7663)

    /// Resolve the current default output device
    private nonisolated static func defaultOutputDeviceID() -> AudioDeviceID? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var deviceID = AudioDeviceID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)

        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID
        )
        guard status == noErr, deviceID != kAudioObjectUnknown else { return nil }
        return deviceID
    }

    /// Read output volume via CoreAudio (0.0 to 1.0), or nil if unavailable
    private nonisolated static func readCoreAudioVolume() -> Float? {
        guard let deviceID = defaultOutputDeviceID() else { return nil }
        return readCoreAudioVolume(deviceID: deviceID)
    }

    /// Read a specific device's output volume (0.0 to 1.0), or nil if unavailable
    private nonisolated static func readCoreAudioVolume(deviceID: AudioDeviceID) -> Float? {

        var volume = Float32(0)
        var size = UInt32(MemoryLayout<Float32>.size)

        // Prefer the virtual main (master) volume
        var address = AudioObjectPropertyAddress(
            mSelector: virtualMainVolumeSelector,
            mScope: kAudioDevicePropertyScopeOutput,
            mElement: kAudioObjectPropertyElementMain
        )
        if AudioObjectHasProperty(deviceID, &address),
           AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &volume) == noErr {
            return max(0.0, min(1.0, volume))
        }

        // Fall back to channel 1's scalar volume
        address.mSelector = kAudioDevicePropertyVolumeScalar
        address.mElement = 1
        size = UInt32(MemoryLayout<Float32>.size)
        if AudioObjectHasProperty(deviceID, &address),
           AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &volume) == noErr {
            return max(0.0, min(1.0, volume))
        }

        return nil
    }

    /// Write output volume via CoreAudio (0.0 to 1.0), returns false if unavailable
    private nonisolated static func writeCoreAudioVolume(_ volume: Float) -> Bool {
        guard let deviceID = defaultOutputDeviceID() else { return false }
        return writeCoreAudioVolume(volume, deviceID: deviceID)
    }

    /// Write a specific device's output volume (0.0 to 1.0), returns false if unavailable
    private nonisolated static func writeCoreAudioVolume(_ volume: Float, deviceID: AudioDeviceID) -> Bool {

        var clamped = Float32(max(0.0, min(1.0, volume)))
        let size = UInt32(MemoryLayout<Float32>.size)

        // Prefer the virtual main (master) volume
        var address = AudioObjectPropertyAddress(
            mSelector: virtualMainVolumeSelector,
            mScope: kAudioDevicePropertyScopeOutput,
            mElement: kAudioObjectPropertyElementMain
        )
        if isPropertySettable(deviceID, &address),
           AudioObjectSetPropertyData(deviceID, &address, 0, nil, size, &clamped) == noErr {
            return true
        }

        // Fall back to per-channel scalar volume (channels 1 and 2)
        var didSet = false
        for channel: UInt32 in 1...2 {
            address.mSelector = kAudioDevicePropertyVolumeScalar
            address.mElement = channel
            if isPropertySettable(deviceID, &address),
               AudioObjectSetPropertyData(deviceID, &address, 0, nil, size, &clamped) == noErr {
                didSet = true
            }
        }
        return didSet
    }

    /// Nominal sample rate of a device — stands in for its active Bluetooth
    /// profile (A2DP runs at 44.1/48 kHz, HFP at 16/24 kHz)
    private nonisolated static func nominalSampleRate(deviceID: AudioDeviceID) -> Float64 {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyNominalSampleRate,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var rate = Float64(0)
        var size = UInt32(MemoryLayout<Float64>.size)
        guard AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &rate) == noErr else { return 0 }
        return rate
    }

    /// Check that a property exists and is settable on the device
    private nonisolated static func isPropertySettable(
        _ deviceID: AudioDeviceID,
        _ address: inout AudioObjectPropertyAddress
    ) -> Bool {
        guard AudioObjectHasProperty(deviceID, &address) else { return false }
        var settable = DarwinBoolean(false)
        let status = AudioObjectIsPropertySettable(deviceID, &address, &settable)
        return status == noErr && settable.boolValue
    }
}
