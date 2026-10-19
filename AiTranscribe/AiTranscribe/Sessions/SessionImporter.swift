import Foundation
import AVFoundation

// MARK: - Session Import

/// Errors surfaced to the import UI.
enum SessionImportError: LocalizedError {
    case noAudioTrack
    case unreadableFile
    case conversionFailed(String)

    var errorDescription: String? {
        switch self {
        case .noAudioTrack:
            return "This file has no audio track."
        case .unreadableFile:
            return "This file format can't be read. Try MP3, WAV, M4A, MP4 or MOV."
        case .conversionFailed(let message):
            return "Audio conversion failed: \(message)"
        }
    }
}

/// A probed media file the user picked for import.
struct ImportableAudioFile: Equatable {
    let url: URL
    let fileSize: Int64
    let duration: TimeInterval

    var fileName: String { url.lastPathComponent }
    var suggestedSessionName: String { url.deletingPathExtension().lastPathComponent }
    var fileExtension: String { url.pathExtension.uppercased() }
}

extension SessionManager {

    /// Probe a media file and confirm it carries a readable audio track.
    nonisolated static func probeImportableAudio(at url: URL) async throws -> ImportableAudioFile {
        let asset = AVURLAsset(url: url)

        let tracks: [AVAssetTrack]
        let duration: CMTime
        do {
            tracks = try await asset.loadTracks(withMediaType: .audio)
            duration = try await asset.load(.duration)
        } catch {
            throw SessionImportError.unreadableFile
        }

        guard !tracks.isEmpty else {
            throw SessionImportError.noAudioTrack
        }

        let seconds = CMTimeGetSeconds(duration)
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        return ImportableAudioFile(
            url: url,
            fileSize: (attrs?[.size] as? Int64) ?? 0,
            duration: seconds.isFinite && seconds > 0 ? seconds : 0
        )
    }

    /// Import any audio/video file as a new session: convert it to the canonical
    /// 16 kHz mono AAC `audio.m4a`, then register a fully formed session so both
    /// transcription routes work on it exactly like a recorded one.
    ///
    /// Conversion happens to a temp file first — a failed import never leaves a
    /// half-built session behind.
    func importSession(
        from sourceURL: URL,
        name: String,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> Session {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("AiTranscribeImport-\(UUID().uuidString).m4a")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let durationSeconds = try await Self.convertToSessionAudio(
            inputURL: sourceURL,
            outputURL: tempURL,
            progress: progress
        )

        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        var session = createSession(name: trimmedName.isEmpty ? nil : trimmedName)
        let audioURL = getAudioURL(for: session)

        do {
            try FileManager.default.moveItem(at: tempURL, to: audioURL)
        } catch {
            deleteSession(id: session.id)
            throw SessionImportError.conversionFailed(error.localizedDescription)
        }

        session.hasAudio = true
        session.duration = durationSeconds
        if let attrs = try? FileManager.default.attributesOfItem(atPath: audioURL.path) {
            session.fileSize = (attrs[.size] as? Int64) ?? 0
        }
        updateSession(session)

        print("SessionManager: Imported '\(sourceURL.lastPathComponent)' as session '\(session.name)', " +
              "duration: \(Self.formatDuration(session.duration)), " +
              "size: \(Self.formatFileSize(session.fileSize))")
        return session
    }

    /// Convert an arbitrary media file to the session format (16 kHz mono AAC in
    /// an .m4a container — the same settings `SessionRecorder` exports). Multiple
    /// audio tracks (e.g. in a screen recording) are mixed down together.
    /// Returns the written audio duration in seconds.
    private nonisolated static func convertToSessionAudio(
        inputURL: URL,
        outputURL: URL,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> TimeInterval {
        let asset = AVURLAsset(url: inputURL)

        let audioTracks: [AVAssetTrack]
        let assetDuration: CMTime
        do {
            audioTracks = try await asset.loadTracks(withMediaType: .audio)
            assetDuration = try await asset.load(.duration)
        } catch {
            throw SessionImportError.unreadableFile
        }

        guard !audioTracks.isEmpty else {
            throw SessionImportError.noAudioTrack
        }

        let totalSeconds = CMTimeGetSeconds(assetDuration)

        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw SessionImportError.unreadableFile
        }

        let readerOutput = AVAssetReaderAudioMixOutput(
            audioTracks: audioTracks,
            audioSettings: [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: 16000,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsNonInterleaved: false,
            ]
        )
        guard reader.canAdd(readerOutput) else {
            throw SessionImportError.unreadableFile
        }
        reader.add(readerOutput)

        try? FileManager.default.removeItem(at: outputURL)
        let writer: AVAssetWriter
        do {
            writer = try AVAssetWriter(outputURL: outputURL, fileType: .m4a)
        } catch {
            throw SessionImportError.conversionFailed(error.localizedDescription)
        }

        let writerInput = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: 16000,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
        ])
        writerInput.expectsMediaDataInRealTime = false
        writer.add(writerInput)

        reader.startReading()
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let queue = DispatchQueue(label: "com.aitranscribe.sessionimport")
            writerInput.requestMediaDataWhenReady(on: queue) {
                while writerInput.isReadyForMoreMediaData {
                    if let sampleBuffer = readerOutput.copyNextSampleBuffer() {
                        writerInput.append(sampleBuffer)

                        if totalSeconds > 0 {
                            let pts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
                            if pts.isFinite {
                                progress(min(1.0, max(0.0, pts / totalSeconds)))
                            }
                        }
                    } else {
                        writerInput.markAsFinished()

                        if reader.status == .failed {
                            writer.cancelWriting()
                            continuation.resume(throwing: SessionImportError.conversionFailed(
                                reader.error?.localizedDescription ?? "Could not read audio data."
                            ))
                            return
                        }

                        writer.finishWriting {
                            if writer.status == .completed {
                                progress(1.0)
                                continuation.resume()
                            } else {
                                continuation.resume(throwing: SessionImportError.conversionFailed(
                                    writer.error?.localizedDescription ?? "Could not write audio file."
                                ))
                            }
                        }
                        return
                    }
                }
            }
        }

        // Prefer the written file's own duration; fall back to the source asset's.
        let writtenAsset = AVURLAsset(url: outputURL)
        if let writtenDuration = try? await writtenAsset.load(.duration) {
            let seconds = CMTimeGetSeconds(writtenDuration)
            if seconds.isFinite && seconds > 0 { return seconds }
        }
        return totalSeconds.isFinite && totalSeconds > 0 ? totalSeconds : 0
    }
}
