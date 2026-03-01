import SwiftUI

struct StatusView: View {
    @EnvironmentObject var appState: AppState
    @State private var pulseAnim = false

    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 0) {
                // Header
                headerBar

                VStack(spacing: 14) {
                    // Breathing status card
                    breathingStatusCard

                    // Metrics grid
                    metricsGrid

                    // CO2 monitoring section
                    co2Section

                    // Patch health
                    patchHealthSection

                    // Emergency button
                    emergencyButton
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 110)
            }
        }
    }

    // MARK: - Header
    var headerBar: some View {
        HStack(spacing: 10) {
            // Mascot icon
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(colors: [Theme.cyan, Theme.violet],
                                       startPoint: .topLeading,
                                       endPoint: .bottomTrailing)
                    )
                    .frame(width: 36, height: 36)
                Image(systemName: "moon.zzz.fill")
                    .font(.system(size: 18))
                    .foregroundColor(.white)
            }

            Text("SleepGuardian")
                .font(.system(size: 20, weight: .semibold))
                .foregroundColor(Theme.violet)

            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.top, 16)
        .padding(.bottom, 14)
    }

    // MARK: - Breathing Status Card
    var breathingStatusCard: some View {
        HStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [Theme.cyan.opacity(0.25), Theme.green.opacity(0.15)],
                            startPoint: .topLeading, endPoint: .bottomTrailing)
                    )
                    .frame(width: 60, height: 60)

                if pulseAnim {
                    Circle()
                        .stroke(Theme.cyan.opacity(0.3), lineWidth: 2)
                        .frame(width: 70, height: 70)
                        .scaleEffect(pulseAnim ? 1.2 : 1.0)
                        .opacity(pulseAnim ? 0 : 0.6)
                }

                Image(systemName: appState.breathingStatus.icon)
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [Theme.cyan, Theme.green],
                            startPoint: .topLeading, endPoint: .bottomTrailing)
                    )
            }
            .onAppear {
                withAnimation(.easeInOut(duration: 2).repeatForever(autoreverses: false)) {
                    pulseAnim = true
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(appState.breathingStatus.title)
                    .font(.system(size: 20, weight: .bold))
                    .foregroundColor(Theme.text)

                Text(appState.breathingStatus.subtitle)
                    .font(.system(size: 13))
                    .foregroundColor(Theme.subtext)
                    .lineSpacing(2)
                    .multilineTextAlignment(.leading)
            }

            Spacer()
        }
        .cardStyle()
    }

    // MARK: - Metrics Grid
    var metricsGrid: some View {
        HStack(spacing: 10) {
            MetricCard(icon: "lungs.fill", iconColor: Theme.cyan,
                       label: "Respiratory\nRate", value: "Normal")
            MetricCard(icon: "arrow.up.and.down.circle.fill", iconColor: Theme.indigo,
                       label: "Chest\nExpansion", value: "Normal")
            MetricCard(icon: "waveform.path.ecg", iconColor: Theme.green,
                       label: "Breathing\nRegularity", value: "Stable")
        }
    }

    // MARK: - CO2 Section
    var co2Section: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("CO₂ Monitoring")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(Theme.text)
                Spacer()
                Image(systemName: "info.circle")
                    .foregroundColor(Theme.subtext)
            }

            // Active indicator card
            HStack(spacing: 14) {
                ZStack {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Theme.violet.opacity(0.15))
                        .frame(width: 48, height: 56)
                    VStack(spacing: 2) {
                        Image(systemName: "thermometer.medium")
                            .font(.system(size: 22))
                            .foregroundColor(Theme.violet)
                        Circle()
                            .fill(Theme.violet)
                            .frame(width: 6, height: 6)
                    }
                }

                VStack(alignment: .leading, spacing: 3) {
                    Text("CO₂ Active")
                        .font(.system(size: 16, weight: .bold))
                        .foregroundColor(Theme.violet)
                    Text("Patch 38°C · Rising ↗")
                        .font(.system(size: 13))
                        .foregroundColor(Theme.subtext)
                }

                Spacer()
                Image(systemName: "chevron.right")
                    .foregroundColor(Theme.subtext)
                    .font(.system(size: 14, weight: .semibold))
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Theme.violet.opacity(0.08))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Theme.violet.opacity(0.3), lineWidth: 0.5)
                    )
            )

            // Status bar: Ambient → Warming → CO₂ Active
            HStack(spacing: 0) {
                statusPill(label: "Ambient", color: Theme.cyan, filled: false)
                connectorLine
                statusPill(label: "Warming", color: Theme.amber, filled: false)
                connectorLine
                statusPill(label: "CO₂ Active", color: Theme.violet, filled: true)
            }
        }
        .cardStyle()
    }

    var connectorLine: some View {
        Rectangle()
            .fill(Theme.dim.opacity(0.5))
            .frame(maxWidth: .infinity, maxHeight: 1)
    }

    func statusPill(label: String, color: Color, filled: Bool) -> some View {
        HStack(spacing: 4) {
            Circle()
                .fill(filled ? color : color.opacity(0.4))
                .frame(width: 8, height: 8)
            Text(label)
                .font(.system(size: 11, weight: filled ? .semibold : .regular))
                .foregroundColor(filled ? color : Theme.subtext)
        }
        .padding(.horizontal, 4)
    }

    // MARK: - Patch Health
    var patchHealthSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Patch Health")
                .font(.system(size: 16, weight: .semibold))
                .foregroundColor(Theme.text)

            HStack(spacing: 10) {
                PatchHealthTile(
                    icon: "battery.75percent",
                    iconColor: Theme.green,
                    label: "Battery",
                    value: "78%"
                )
                PatchHealthTile(
                    icon: "wifi",
                    iconColor: Theme.cyan,
                    label: "Connected",
                    value: nil
                )
                PatchHealthTile(
                    icon: "hand.raised.fingers.spread.fill",
                    iconColor: Theme.violet,
                    label: "Good Contact",
                    value: nil
                )
            }
        }
        .cardStyle()
    }

    // MARK: - Emergency Button
    var emergencyButton: some View {
        Button(action: {}) {
            HStack(spacing: 10) {
                Image(systemName: "phone.fill")
                    .font(.system(size: 18, weight: .semibold))
                Text("Call Emergency Care")
                    .font(.system(size: 17, weight: .semibold))
            }
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 18)
            .background(
                LinearGradient(
                    colors: [Color(hex: "b91c1c"), Color(hex: "7f1d1d")],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
            .clipShape(Capsule())
            .shadow(color: Theme.red.opacity(0.4), radius: 12, y: 6)
        }
    }
}

// MARK: - Sub-components
struct MetricCard: View {
    let icon: String
    let iconColor: Color
    let label: String
    let value: String

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 26))
                .foregroundColor(iconColor)

            Text(label)
                .font(.system(size: 12))
                .foregroundColor(Theme.subtext)
                .multilineTextAlignment(.center)
                .lineSpacing(1)

            Text(value)
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(Theme.text)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 16)
        .padding(.horizontal, 8)
        .cardStyle(padding: 0)
    }
}

struct PatchHealthTile: View {
    let icon: String
    let iconColor: Color
    let label: String
    let value: String?

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 22))
                .foregroundColor(iconColor)

            Text(label)
                .font(.system(size: 12))
                .foregroundColor(Theme.subtext)
                .multilineTextAlignment(.center)

            if let value {
                Text(value)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(Theme.text)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Theme.bg3)
        )
    }
}

