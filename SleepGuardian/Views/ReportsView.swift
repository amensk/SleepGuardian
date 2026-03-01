import SwiftUI

struct ReportsView: View {

    let breathingTrend: [DataPoint] = {
        let days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        let vals: [Double] = [0.55, 0.50, 0.48, 0.60, 0.72, 0.80, 0.88]
        return zip(days, vals).enumerated().map { i, pair in
            DataPoint(x: Double(i), y: pair.1, label: pair.0)
        }
    }()

    let co2FreqData: [WeekBarData] = {
        let days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        let vals: [Double] = [0.4, 0.6, 0.5, 0.7, 0.8, 0.65, 0.9]
        return zip(days, vals).map { day, val in
            WeekBarData(day: day, value: val, color: Color(hex: "00e5ff"))
        }
    }()

    let alertsData: [WeekBarData] = {
        let days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        let urgentDays: Set<Int> = [2, 4]
        return days.enumerated().map { i, day in
            let isUrgent = urgentDays.contains(i)
            let val = isUrgent ? Double.random(in: 0.6...0.9) : Double.random(in: 0.1...0.45)
            return WeekBarData(day: day, value: val, color: isUrgent ? Color(hex: "ef4444") : Color(hex: "c084fc"))
        }
    }()

    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 0) {
                // Header
                HStack {
                    ZStack {
                        Circle()
                            .fill(
                                LinearGradient(colors: [Theme.cyan, Theme.violet],
                                               startPoint: .topLeading,
                                               endPoint: .bottomTrailing)
                            )
                            .frame(width: 32, height: 32)
                        Image(systemName: "moon.zzz.fill")
                            .font(.system(size: 16))
                            .foregroundColor(.white)
                    }

                    Text("Reports")
                        .font(.system(size: 22, weight: .bold))
                        .foregroundColor(Theme.text)

                    Spacer()
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, 14)

                VStack(spacing: 14) {
                    // Last night summary
                    lastNightCard

                    // Sleep timeline bar
                    sleepTimelineCard

                    // Weekly insights header
                    HStack {
                        Text("Weekly Insights")
                            .font(.system(size: 17, weight: .bold))
                            .foregroundColor(Theme.text)
                        Spacer()
                    }

                    // Breathing stability trend
                    reportCard(title: "Breathing Stability Trend") {
                        breathingTrendChart
                    }

                    // CO2 monitoring frequency
                    reportCard(title: "CO₂ Monitoring Frequency") {
                        co2FrequencyChart
                    }

                    // Alerts this week
                    alertsCard

                    // Share button
                    Button(action: {}) {
                        HStack(spacing: 10) {
                            Image(systemName: "square.and.arrow.up")
                                .font(.system(size: 16, weight: .semibold))
                            Text("Share with Clinician")
                                .font(.system(size: 16, weight: .semibold))
                        }
                        .foregroundColor(Theme.text)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(
                            RoundedRectangle(cornerRadius: 14)
                                .fill(Color(hex: "1c1435"))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 14)
                                        .stroke(Theme.violet.opacity(0.4), lineWidth: 0.8)
                                )
                        )
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 110)
            }
        }
    }

    // MARK: - Last Night Card
    var lastNightCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "moon.fill")
                    .foregroundColor(Theme.violet)
                Text("Last Night Summary")
                    .font(.system(size: 17, weight: .bold))
                    .foregroundColor(Theme.text)
            }

            HStack(spacing: 6) {
                Image(systemName: "chart.bar.fill")
                    .font(.system(size: 12))
                    .foregroundColor(Theme.cyan)
                Text("Monitored for 7 hr 15 min")
                    .font(.system(size: 14))
                    .foregroundColor(Theme.subtext)
            }

            HStack(spacing: 20) {
                summaryItem(value: "3", label: "Monitoring\nEvents",
                            valueColor: Theme.text)
                Divider()
                    .frame(height: 36)
                    .background(Theme.dim)
                summaryItem(value: "1", label: "Urgent\nAlert",
                            valueColor: Theme.red)
            }
        }
        .cardStyle()
    }

    func summaryItem(value: String, label: String, valueColor: Color) -> some View {
        HStack(spacing: 8) {
            Text(value)
                .font(.system(size: 28, weight: .bold))
                .foregroundColor(valueColor)
            Text(label)
                .font(.system(size: 13))
                .foregroundColor(valueColor == Theme.red ? Theme.red : Theme.subtext)
                .lineSpacing(2)
        }
    }

    // MARK: - Sleep Timeline
    var sleepTimelineCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("9PM")
                    .font(.system(size: 10))
                    .foregroundColor(Theme.subtext)
                Spacer()
                Text("12AM")
                    .font(.system(size: 10))
                    .foregroundColor(Theme.subtext)
                Spacer()
                Text("2:4M")
                    .font(.system(size: 10))
                    .foregroundColor(Theme.subtext)
                Spacer()
                Text("2:4M")
                    .font(.system(size: 10))
                    .foregroundColor(Theme.subtext)
            }

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    // Gradient bar
                    LinearGradient(
                        colors: [
                            Color(hex: "22c55e"),
                            Color(hex: "00e5ff"),
                            Color(hex: "818cf8"),
                            Color(hex: "ef4444"),
                            Color(hex: "818cf8"),
                            Color(hex: "facc15"),
                            Color(hex: "22c55e"),
                        ],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                    .frame(height: 16)
                    .clipShape(Capsule())

                    // Indicator
                    Circle()
                        .fill(Color.white)
                        .frame(width: 22, height: 22)
                        .shadow(color: Color.white.opacity(0.4), radius: 4)
                        .overlay(
                            Circle()
                                .fill(Theme.violet)
                                .frame(width: 10, height: 10)
                        )
                        .offset(x: geo.size.width * 0.82)
                }
            }
            .frame(height: 22)
        }
        .cardStyle()
    }

    // MARK: - Card wrapper
    func reportCard<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.system(size: 15, weight: .semibold))
                .foregroundColor(Theme.text)
            content()
        }
        .cardStyle()
    }

    // MARK: - Breathing Trend Chart
    var breathingTrendChart: some View {
        VStack(spacing: 6) {
            GeometryReader { geo in
                ZStack {
                    smoothLine(data: breathingTrend, size: geo.size, color: Theme.cyan)

                    ForEach(breathingTrend.indices, id: \.self) { i in
                        let pt = chartPoint(breathingTrend[i], in: breathingTrend, size: geo.size)
                        Circle()
                            .fill(i == breathingTrend.count - 1 ? Theme.violet : Theme.cyan)
                            .frame(width: i == breathingTrend.count - 1 ? 9 : 6,
                                   height: i == breathingTrend.count - 1 ? 9 : 6)
                            .position(pt)
                    }
                }
            }
            .frame(height: 90)

            HStack {
                ForEach(breathingTrend, id: \.id) { dp in
                    Text(dp.label)
                        .font(.system(size: 9))
                        .foregroundColor(Theme.subtext)
                        .frame(maxWidth: .infinity)
                }
            }
        }
    }

    // MARK: - CO2 Frequency Chart
    var co2FrequencyChart: some View {
        VStack(spacing: 6) {
            GeometryReader { geo in
                HStack(alignment: .bottom, spacing: 6) {
                    ForEach(co2FreqData) { dp in
                        VStack {
                            Spacer()
                            if dp == co2FreqData.last {
                                Image(systemName: "arrow.up.right")
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundColor(Theme.cyan)
                                    .offset(y: -4)
                            }
                            RoundedRectangle(cornerRadius: 4)
                                .fill(
                                    LinearGradient(
                                        colors: [dp.color, dp.color.opacity(0.4)],
                                        startPoint: .top, endPoint: .bottom)
                                )
                                .frame(maxWidth: .infinity)
                                .frame(height: geo.size.height * dp.value)
                        }
                    }
                }
            }
            .frame(height: 80)

            HStack {
                ForEach(co2FreqData, id: \.id) { dp in
                    Text(dp.day)
                        .font(.system(size: 9))
                        .foregroundColor(Theme.subtext)
                        .frame(maxWidth: .infinity)
                }
            }
        }
    }

    // MARK: - Alerts Card
    var alertsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(Theme.amber)
                    .font(.system(size: 14))
                Text("Alerts This Week")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(Theme.text)
                Spacer()
                Text("3 Total")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(Theme.text)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 5)
                    .background(
                        Capsule()
                            .fill(Theme.bg3)
                    )
            }

            GeometryReader { geo in
                HStack(alignment: .bottom, spacing: 6) {
                    ForEach(alertsData) { dp in
                        RoundedRectangle(cornerRadius: 4)
                            .fill(
                                LinearGradient(
                                    colors: [dp.color, dp.color.opacity(0.5)],
                                    startPoint: .top, endPoint: .bottom)
                            )
                            .frame(maxWidth: .infinity)
                            .frame(height: geo.size.height * dp.value)
                    }
                }
                .frame(maxHeight: .infinity, alignment: .bottom)
            }
            .frame(height: 70)

            HStack {
                ForEach(alertsData, id: \.id) { dp in
                    Text(dp.day)
                        .font(.system(size: 9))
                        .foregroundColor(Theme.subtext)
                        .frame(maxWidth: .infinity)
                }
            }
        }
        .cardStyle()
    }

    // MARK: - Chart helpers
    func chartPoint(_ dp: DataPoint, in data: [DataPoint], size: CGSize) -> CGPoint {
        let minX = data.map(\.x).min() ?? 0
        let maxX = data.map(\.x).max() ?? 1
        let xRange = maxX - minX == 0 ? 1 : maxX - minX
        let x = (dp.x - minX) / xRange * size.width
        let y = (1 - dp.y) * size.height
        return CGPoint(x: x, y: y)
    }

    func smoothLine(data: [DataPoint], size: CGSize, color: Color) -> some View {
        Canvas { ctx, canvasSize in
            guard data.count > 1 else { return }
            let pts = data.map { chartPoint($0, in: data, size: canvasSize) }
            var path = Path()
            path.move(to: pts[0])
            for i in 1..<pts.count {
                let cp1 = CGPoint(x: (pts[i-1].x + pts[i].x) / 2, y: pts[i-1].y)
                let cp2 = CGPoint(x: (pts[i-1].x + pts[i].x) / 2, y: pts[i].y)
                path.addCurve(to: pts[i], control1: cp1, control2: cp2)
            }
            ctx.stroke(path, with: .color(color), lineWidth: 2.5)

            var fill = path
            fill.addLine(to: CGPoint(x: pts.last!.x, y: canvasSize.height))
            fill.addLine(to: CGPoint(x: pts.first!.x, y: canvasSize.height))
            fill.closeSubpath()
            ctx.fill(fill, with: .color(color.opacity(0.12)))
        }
    }
}

// WeekBarData equatable for last check
extension WeekBarData: Equatable {
    static func == (lhs: WeekBarData, rhs: WeekBarData) -> Bool {
        lhs.id == rhs.id
    }
}
