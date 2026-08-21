import SwiftUI

/// The mixer shown in the menu bar while a session is recording.
///
/// Two channel strips, each a meter and a fader. The meter reads post-fader,
/// so what you see is what is going into the file, and moving the fader moves
/// the bar. That feedback loop is the point: the level is set by watching it,
/// not by an algorithm guessing what it should be.
struct SessionMixerView: View {
    @ObservedObject var recorder: SessionRecorder

    var body: some View {
        VStack(spacing: 10) {
            channel(
                title: "Microphone",
                icon: "mic.fill",
                level: recorder.micLevel,
                gain: $recorder.micGainDB
            )
            channel(
                title: "Desktop Audio",
                icon: "speaker.wave.2.fill",
                level: recorder.systemLevel,
                gain: $recorder.systemGainDB
            )
        }
    }

    private func channel(
        title: String,
        icon: String,
        level: AudioLevelReading,
        gain: Binding<Double>
    ) -> some View {
        VStack(spacing: 5) {
            HStack(spacing: 5) {
                Image(systemName: icon)
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
                Text(title)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                Spacer()
                Text(levelText(level))
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                    .foregroundStyle(colour(for: level.zone))
                if gain.wrappedValue > 0.25 {
                    Image(systemName: "bolt.fill")
                        .font(.system(size: 8))
                        .foregroundStyle(boostColour(gain.wrappedValue))
                }
                Text(gainText(gain.wrappedValue))
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                    .foregroundStyle(gain.wrappedValue > 0.25 ? boostColour(gain.wrappedValue) : Color.secondary)
                    .frame(width: 42, alignment: .trailing)
            }

            MeterBar(level: level)
                .frame(height: 6)

            // Range reaches +40 dB because a call app can drop the microphone
            // by 25-30 dB, and boosting is the only lever left once it has.
            // The limiter downstream means the top of the range cannot clip.
            Slider(value: gain, in: -40...40, step: 0.5)
                .controlSize(.mini)
                .tint(gain.wrappedValue > 0.25 ? boostColour(gain.wrappedValue) : Color.accentColor)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 7))
    }

    private func levelText(_ level: AudioLevelReading) -> String {
        level.magnitudeDB <= AudioLevelMeter.minimumDB
            ? "—"
            : String(format: "%.0f dB", level.magnitudeDB)
    }

    private func gainText(_ db: Double) -> String {
        abs(db) < 0.25 ? "0.0" : String(format: "%+.1f", db)
    }

    /// Boost is worth showing: above unity the fader is amplifying the noise
    /// floor along with the voice, and past about +20 dB that starts to be
    /// audible. Amber for boosting, red once it is heavy.
    private func boostColour(_ db: Double) -> Color {
        db >= 20 ? Color(MeterArcStyle.litError) : Color(MeterArcStyle.litWarning)
    }

    private func colour(for zone: AudioLevelZone) -> Color {
        switch zone {
        case .nominal: return Color(MeterArcStyle.litNominal)
        case .warning: return Color(MeterArcStyle.litWarning)
        case .error: return Color(MeterArcStyle.litError)
        }
    }
}

/// A horizontal meter using the same scale and colours as the arcs on the
/// indicator, so the two always agree.
struct MeterBar: View {
    let level: AudioLevelReading

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let warningStart = CGFloat(MeterArcStyle.warningStart)
            let errorStart = CGFloat(MeterArcStyle.errorStart)
            let filled = CGFloat(level.fraction)

            ZStack(alignment: .leading) {
                // Unlit zones, so the shape of the scale is visible in silence.
                HStack(spacing: 0) {
                    Color(MeterArcStyle.dimNominal).frame(width: width * warningStart)
                    Color(MeterArcStyle.dimWarning).frame(width: width * (errorStart - warningStart))
                    Color(MeterArcStyle.dimError)
                }
                .opacity(0.45)

                // Lit portion, clipped per zone so green stays green when the
                // tip goes yellow.
                HStack(spacing: 0) {
                    Color(MeterArcStyle.litNominal)
                        .frame(width: width * min(filled, warningStart))
                    Color(MeterArcStyle.litWarning)
                        .frame(width: width * max(0, min(filled, errorStart) - warningStart))
                    Color(MeterArcStyle.litError)
                        .frame(width: width * max(0, min(filled, 1) - errorStart))
                    Spacer(minLength: 0)
                }

                // Held peak.
                if level.peakHoldFraction > 0.02 {
                    Rectangle()
                        .fill(.white.opacity(0.85))
                        .frame(width: 1.5)
                        .offset(x: width * CGFloat(level.peakHoldFraction) - 1.5)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 3))
        }
    }
}
