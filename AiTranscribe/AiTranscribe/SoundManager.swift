/*
 SoundManager.swift
 ==================

 Manages sound effects for recording start/stop.
 Uses custom mp3 sound files bundled with the app.
 */

import AVFoundation
import AppKit

/// Manages sound effects for the app
class SoundManager {
    /// Singleton instance
    static let shared = SoundManager()

    /// Audio players for each sound
    private var startPlayer: AVAudioPlayer?
    private var stopPlayer: AVAudioPlayer?

    private init() {
        loadSounds()
    }

    /// Load sound files from bundle
    private func loadSounds() {
        // Load start sound (on.mp3)
        if let url = Bundle.main.url(forResource: "on", withExtension: "mp3") {
            do {
                startPlayer = try AVAudioPlayer(contentsOf: url)
                startPlayer?.prepareToPlay()
                print("Loaded on.mp3 from bundle")
            } catch {
                print("Failed to load on.mp3: \(error)")
            }
        } else {
            print("on.mp3 not found in bundle - make sure to add it to Xcode project")
        }

        // Load stop sound (off.mp3)
        if let url = Bundle.main.url(forResource: "off", withExtension: "mp3") {
            do {
                stopPlayer = try AVAudioPlayer(contentsOf: url)
                stopPlayer?.prepareToPlay()
                print("Loaded off.mp3 from bundle")
            } catch {
                print("Failed to load off.mp3: \(error)")
            }
        } else {
            print("off.mp3 not found in bundle - make sure to add it to Xcode project")
        }
    }

    /// Check if sounds are enabled in settings
    private var soundsEnabled: Bool {
        // Default to true if not yet set (first launch)
        if UserDefaults.standard.object(forKey: "playSounds") == nil {
            return true
        }
        return UserDefaults.standard.bool(forKey: "playSounds")
    }

    /// Play start recording sound
    func playStartSound() {
        guard soundsEnabled else { return }
        startPlayer?.currentTime = 0
        startPlayer?.play()
    }

    /// Play stop recording sound
    func playStopSound() {
        guard soundsEnabled else { return }
        stopPlayer?.currentTime = 0
        stopPlayer?.play()
    }
}
