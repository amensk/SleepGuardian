import SwiftUI
import Combine

// MARK: - Alert Model
struct SleepAlert: Identifiable {
    let id = UUID()
    let severity: AlertSeverity
    let title: String
    let description: String
    let indicators: [String]
    let timestamp: String

    enum AlertSeverity {
        case urgent, monitoring
        var color: Color {
            switch self {
            case .urgent: return Color(hex: "ef4444")
            case .monitoring: return Color(hex: "facc15")
            }
        }
        var label: String {
            switch self {
            case .urgent: return "URGENT"
            case .monitoring: return "Monitoring"
            }
        }
    }
}

// MARK: - Chart Data
struct DataPoint: Identifiable {
    let id = UUID()
    let x: Double
    let y: Double
    let label: String
}

struct WeekBarData: Identifiable {
    let id = UUID()
    let day: String
    let value: Double
    let color: Color
}

// MARK: - App State
class AppState: ObservableObject {
    @Published var breathingStatus: BreathingStatus = .stable
    @Published var co2Status: CO2Status = .active(temp: 38)
    @Published var batteryLevel: Int = 78
    @Published var isConnected: Bool = true
    @Published var hasGoodContact: Bool = true
    @Published var alerts: [SleepAlert] = SleepAlert.sampleAlerts

    static let shared = AppState()

    enum BreathingStatus {
        case stable, warning, critical
        var title: String {
            switch self {
            case .stable: return "Breathing Stable"
            case .warning: return "Irregular Breathing"
            case .critical: return "Critical Alert"
            }
        }
        var subtitle: String {
            switch self {
            case .stable: return "Breathing rate and depth\nare within normal range"
            case .warning: return "Breathing rate is slightly\nbelow normal range"
            case .critical: return "Breathing appears\ndangerously inadequate"
            }
        }
        var color: Color {
            switch self {
            case .stable: return Color(hex: "00e5ff")
            case .warning: return Color(hex: "facc15")
            case .critical: return Color(hex: "ef4444")
            }
        }
        var icon: String {
            switch self {
            case .stable: return "checkmark.shield.fill"
            case .warning: return "exclamationmark.triangle.fill"
            case .critical: return "exclamationmark.shield.fill"
            }
        }
    }

    enum CO2Status {
        case ambient, warming, active(temp: Int)
        var label: String {
            switch self {
            case .ambient: return "Ambient"
            case .warming: return "Warming"
            case .active(let t): return "CO₂ Active"
            }
        }
        var detail: String {
            switch self {
            case .ambient: return "Sensor ready"
            case .warming: return "Heating up..."
            case .active(let t): return "Patch \(t)°C · Rising ↗"
            }
        }
    }
}

extension SleepAlert {
    static let sampleAlerts: [SleepAlert] = [
        SleepAlert(
            severity: .monitoring,
            title: "Monitoring",
            description: "Breathing shallow and irregular",
            indicators: ["↓ RR", "↓ Expansion"],
            timestamp: "2:30 AM"
        ),
        SleepAlert(
            severity: .monitoring,
            title: "Monitoring",
            description: "Suspicious Breathing Episode",
            indicators: ["↓ RR", "↓ Expansion", "↑ Irregularity"],
            timestamp: "12:15 AM"
        ),
        SleepAlert(
            severity: .urgent,
            title: "Urgent",
            description: "Severe Hypoventilation",
            indicators: ["↓ RR", "↓ Expansion", "↑ Irregularity", "↑ CO₂"],
            timestamp: "Yesterday · 3:10 AM"
        ),
    ]
}

// MARK: - Color Extension
extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3:
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6:
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8:
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0)
        }
        self.init(.sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255)
    }
}
