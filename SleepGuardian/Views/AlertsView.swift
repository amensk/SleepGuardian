import SwiftUI

struct AlertsView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 0) {
                // Header
                HStack {
                    Text("Alerts")
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(Theme.text)
                    Spacer()
                    ZStack(alignment: .topTrailing) {
                        Image(systemName: "bell.fill")
                            .font(.system(size: 22))
                            .foregroundColor(Theme.violet)
                        Circle()
                            .fill(Theme.red)
                            .frame(width: 10, height: 10)
                            .offset(x: 4, y: -3)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, 20)

                VStack(spacing: 14) {
                    // Urgent banner at top
                    urgentBanner

                    // Follow emergency plan button
                    followPlanButton

                    // Past Alerts section
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Past Alerts")
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundColor(Theme.violet)
                            .padding(.leading, 4)

                        ForEach(appState.alerts) { alert in
                            AlertRow(alert: alert)
                        }
                    }

                    // Emergency button
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
                .padding(.horizontal, 16)
                .padding(.bottom, 110)
            }
        }
    }

    // MARK: - Urgent Banner
    var urgentBanner: some View {
        HStack(alignment: .top, spacing: 14) {
            ZStack {
                Circle()
                    .fill(Theme.red.opacity(0.2))
                    .frame(width: 44, height: 44)
                Image(systemName: "exclamationmark.shield.fill")
                    .font(.system(size: 24))
                    .foregroundColor(Theme.red)
            }

            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 6) {
                    Text("URGENT")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(Theme.red)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(Theme.red.opacity(0.15))
                        .clipShape(Capsule())

                    Spacer()

                    Image(systemName: "chevron.right")
                        .foregroundColor(Theme.subtext)
                        .font(.system(size: 14))
                }

                Text("Attention Needed")
                    .font(.system(size: 18, weight: .bold))
                    .foregroundColor(Theme.red)

                Text("Breathing appears dangerously inadequate")
                    .font(.system(size: 13))
                    .foregroundColor(Theme.subtext)
                    .lineSpacing(2)

                HStack(spacing: 8) {
                    ForEach(["↓ RR", "↓ Expansion", "↑ Irregularity"], id: \.self) { indicator in
                        Text(indicator)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundColor(Theme.red.opacity(0.85))
                    }
                }
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(hex: "1a0a0a"))
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Theme.red.opacity(0.4), lineWidth: 0.8)
                )
        )
    }

    // MARK: - Follow Plan
    var followPlanButton: some View {
        Button(action: {}) {
            HStack {
                Text("Follow emergency plan now")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(Theme.text)
                Spacer()
                Image(systemName: "chevron.right")
                    .foregroundColor(Theme.subtext)
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 16)
            .background(
                RoundedRectangle(cornerRadius: 14)
                    .fill(Color(hex: "1c1435"))
                    .overlay(
                        RoundedRectangle(cornerRadius: 14)
                            .stroke(Theme.violet.opacity(0.3), lineWidth: 0.5)
                    )
            )
        }
    }
}

// MARK: - Alert Row
struct AlertRow: View {
    let alert: SleepAlert

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            ZStack {
                Circle()
                    .fill(alert.severity.color.opacity(0.15))
                    .frame(width: 36, height: 36)
                Image(systemName: alert.severity == .urgent ? "exclamationmark.circle.fill" : "exclamationmark.triangle.fill")
                    .font(.system(size: 18))
                    .foregroundColor(alert.severity.color)
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(alert.title)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(alert.severity.color)
                    Spacer()
                    Text(alert.timestamp)
                        .font(.system(size: 11))
                        .foregroundColor(Theme.subtext)
                }

                Text(alert.description)
                    .font(.system(size: 13))
                    .foregroundColor(Theme.text)

                HStack(spacing: 8) {
                    ForEach(alert.indicators, id: \.self) { ind in
                        HStack(spacing: 3) {
                            Image(systemName: ind.hasPrefix("↓") ? "lungs.fill" : "arrow.up.circle.fill")
                                .font(.system(size: 9))
                                .foregroundColor(alert.severity.color.opacity(0.7))
                            Text(ind)
                                .font(.system(size: 11, weight: .medium))
                                .foregroundColor(Theme.subtext)
                        }
                    }
                }
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Theme.bg2)
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(alert.severity.color.opacity(0.2), lineWidth: 0.5)
                )
        )
    }
}
