import SwiftUI

struct TrendsView: View {
    @State private var selectedRange: Int = 2  // 0=6h, 1=12h, 2=24h
    let ranges = ["6h", "12h", "24h"]

    // Respiratory rate data (normalized 0–1)
    let rrData: [DataPoint] = {
        let labels = ["12AM", "4AM", "8AM", "12PM", "4PM", "6PM"]
        let vals: [Double] = [0.55, 0.38, 0.30, 0.45, 0.75, 0.90]
        return zip(labels, vals).enumerated().map { i, pair in
            DataPoint(x: Double(i), y: pair.1, label: pair.0)
        }
    }()

    // Chest expansion bars
    let chestData: [DataPoint] = (0..<40).map { i in
        DataPoint(x: Double(i), y: Double.random(in: 0.2...0.95), label: "")
    }

    // Regularity index
    let regularityData: [DataPoint] = {
        let xs = stride(from: 0.0, to: 1.0, by: 0.05).map { $0 }
        return xs.enumerated().map { i, x in
            let y = 0.3 + 0.4 * sin(x * .pi * 3) + Double.random(in: -0.05...0.05)
            return DataPoint(x: x, y: y.clamped(to: 0...1), label: "")
        }
    }()

    // CO2 activation bars
    let co2Data: [WeekBarData] = {
        let segments: [(color: Color, count: Int)] = [
            (Color(hex: "00e5ff"), 3),
            (Color(hex: "facc15"), 2),
            (Color(hex: "c084fc"), 4),
            (Color(hex: "facc15"), 1),
            (Color(hex: "00e5ff"), 2),
            (Color(hex: "c084fc"), 3),
        ]
        return segments.enumerated().flatMap { i, seg in
            (0..<seg.count).map { j in
                WeekBarData(day: "\(i)-\(j)", value: Double.random(in: 0.3...0.85), color: seg.color)
            }
        }
    }()

    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 0) {
                // Header
                HStack {
                    Image(systemName: "waveform.path.ecg")
                        .foregroundColor(Theme.cyan)
                    Text("Trends")
                        .font(.system(size: 22, weight: .bold))
                        .foregroundColor(Theme.text)
                    Spacer()
                    Image(systemName: "waveform.path.ecg")
                        .foregroundColor(Theme.cyan)
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, 14)

                VStack(spacing: 14) {
                    // Range selector
                    rangeSelector

                    // RR Chart
                    trendCard(title: "Respiratory Rate vs Baseline") {
                        rrChart
                    }

                    // Chest expansion
                    trendCard(title: "Chest Expansion (Depth)") {
                        chestExpansionChart
                    }

                    // Regularity index
                    trendCard(title: "Breathing Regularity Index") {
                        regularityChart
                    }

                    // CO2 activation
                    trendCard(title: "CO₂ Monitoring Activation") {
                        co2ActivationChart
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 110)
            }
        }
    }

    // MARK: - Range Selector
    var rangeSelector: some View {
        HStack(spacing: 0) {
            ForEach(ranges.indices, id: \.self) { i in
                Button(action: { selectedRange = i }) {
                    Text(ranges[i])
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(selectedRange == i ? .white : Theme.subtext)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(
                            RoundedRectangle(cornerRadius: 10)
                                .fill(selectedRange == i ? Theme.violet.opacity(0.6) : Color.clear)
                        )
                }
            }
        }
        .padding(4)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Theme.bg2)
        )
    }

    // MARK: - Card wrapper
    func trendCard<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.system(size: 15, weight: .semibold))
                .foregroundColor(Theme.text)
            content()
        }
        .cardStyle()
    }

    // MARK: - RR Chart (custom line chart)
    var rrChart: some View {
        VStack(spacing: 4) {
            GeometryReader { geo in
                ZStack {
                    // Normal range band
                    let bandY1 = geo.size.height * 0.25
                    let bandY2 = geo.size.height * 0.60
                    Rectangle()
                        .fill(Theme.cyan.opacity(0.06))
                        .frame(height: bandY2 - bandY1)
                        .offset(y: bandY1)

                    Text("Normal Range")
                        .font(.system(size: 9))
                        .foregroundColor(Theme.cyan.opacity(0.5))
                        .offset(x: 4, y: bandY1 + 2)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)

                    // Line
                    lineChart(data: rrData, size: geo.size, color: Theme.cyan)

                    // Dots
                    ForEach(rrData.indices, id: \.self) { i in
                        let pt = point(rrData[i], in: rrData, size: geo.size)
                        Circle()
                            .fill(i == rrData.count - 1 ? Color.white : Theme.amber)
                            .frame(width: i == rrData.count - 1 ? 8 : 6, height: i == rrData.count - 1 ? 8 : 6)
                            .position(pt)
                    }
                }
            }
            .frame(height: 110)

            // X labels
            HStack {
                ForEach(rrData, id: \.id) { dp in
                    Text(dp.label)
                        .font(.system(size: 9))
                        .foregroundColor(Theme.subtext)
                        .frame(maxWidth: .infinity)
                }
            }
        }
    }

    // MARK: - Chest Expansion
    var chestExpansionChart: some View {
        VStack(spacing: 4) {
            GeometryReader { geo in
                HStack(alignment: .bottom, spacing: 2) {
                    ForEach(chestData) { dp in
                        Rectangle()
                            .fill(Theme.cyan.opacity(0.7))
                            .frame(maxWidth: .infinity)
                            .frame(height: geo.size.height * dp.y)
                    }
                }
                .frame(maxHeight: .infinity, alignment: .bottom)

                // "Deep" label
                Text("Deep")
                    .font(.system(size: 9))
                    .foregroundColor(Theme.subtext)
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .offset(y: 0)
            }
            .frame(height: 80)

            HStack {
                Text("Shallow")
                    .font(.system(size: 9))
                    .foregroundColor(Theme.subtext)

                Spacer()

                ForEach(["4AM", "8AM", "12PM", "4PM", "8PM"], id: \.self) { label in
                    Text(label)
                        .font(.system(size: 9))
                        .foregroundColor(Theme.subtext)
                }
            }
        }
    }

    // MARK: - Regularity
    var regularityChart: some View {
        VStack(spacing: 4) {
            GeometryReader { geo in
                ZStack {
                    // Y axis label
                    VStack {
                        Text("100")
                            .font(.system(size: 9))
                            .foregroundColor(Theme.subtext)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Spacer()
                        Text("0")
                            .font(.system(size: 9))
                            .foregroundColor(Theme.subtext)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    // Smooth line
                    smoothLine(data: regularityData, size: geo.size, color: Theme.cyan)
                        .padding(.leading, 20)

                    // End dot
                    if let last = regularityData.last {
                        let pt = point(last, in: regularityData, size: CGSize(width: geo.size.width - 20, height: geo.size.height))
                        Circle()
                            .fill(Theme.cyan)
                            .frame(width: 8, height: 8)
                            .position(x: pt.x + 20, y: pt.y)
                    }
                }
            }
            .frame(height: 90)

            HStack {
                Text("12.4M")
                    .font(.system(size: 9)).foregroundColor(Theme.subtext)
                Text("4.4M")
                    .font(.system(size: 9)).foregroundColor(Theme.subtext)
                Spacer()
                Text("8AM")
                    .font(.system(size: 9)).foregroundColor(Theme.subtext)
                Spacer()
                Text("12PM")
                    .font(.system(size: 9)).foregroundColor(Theme.subtext)
                Spacer()
                Text("4PM")
                    .font(.system(size: 9)).foregroundColor(Theme.subtext)
                Spacer()
                Text("8PM")
                    .font(.system(size: 9)).foregroundColor(Theme.subtext)
            }
        }
    }

    // MARK: - CO2 Activation
    var co2ActivationChart: some View {
        VStack(spacing: 8) {
            GeometryReader { geo in
                HStack(alignment: .center, spacing: 3) {
                    ForEach(co2Data) { dp in
                        Capsule()
                            .fill(dp.color.opacity(0.75))
                            .frame(maxWidth: .infinity)
                            .frame(height: geo.size.height * 0.55)
                    }
                }
                .frame(maxHeight: .infinity, alignment: .center)
            }
            .frame(height: 36)

            HStack(spacing: 16) {
                legendItem(color: Theme.cyan, label: "Ambient")
                legendItem(color: Theme.amber, label: "Warming")
                legendItem(color: Theme.violet, label: "CO₂ Active")
                Spacer()
            }
        }
    }

    func legendItem(color: Color, label: String) -> some View {
        HStack(spacing: 5) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(label)
                .font(.system(size: 11))
                .foregroundColor(Theme.subtext)
        }
    }

    // MARK: - Chart helpers
    func point(_ dp: DataPoint, in data: [DataPoint], size: CGSize) -> CGPoint {
        let minX = data.map(\.x).min() ?? 0
        let maxX = data.map(\.x).max() ?? 1
        let minY = 0.0
        let maxY = 1.0
        let xRange = maxX - minX == 0 ? 1 : maxX - minX
        let x = (dp.x - minX) / xRange * size.width
        let y = (1 - (dp.y - minY) / (maxY - minY)) * size.height
        return CGPoint(x: x, y: y)
    }

    func lineChart(data: [DataPoint], size: CGSize, color: Color) -> some View {
        Canvas { ctx, _ in
            guard data.count > 1 else { return }
            var path = Path()
            let pts = data.map { point($0, in: data, size: size) }
            path.move(to: pts[0])
            for pt in pts.dropFirst() {
                path.addLine(to: pt)
            }
            ctx.stroke(path, with: .color(color), lineWidth: 2)

            // Fill area
            var fill = path
            fill.addLine(to: CGPoint(x: pts.last!.x, y: size.height))
            fill.addLine(to: CGPoint(x: pts.first!.x, y: size.height))
            fill.closeSubpath()
            ctx.fill(fill, with: .color(color.opacity(0.1)))
        }
    }

    func smoothLine(data: [DataPoint], size: CGSize, color: Color) -> some View {
        Canvas { ctx, canvasSize in
            guard data.count > 1 else { return }
            let pts = data.map { point($0, in: data, size: canvasSize) }
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

// MARK: - Comparable clamp
extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
