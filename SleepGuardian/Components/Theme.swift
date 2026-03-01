import SwiftUI

// MARK: - Design Tokens
enum Theme {
    // Background layers
    static let bg       = Color(hex: "0a0718")   // deepest bg
    static let bg2      = Color(hex: "110e24")   // card bg
    static let bg3      = Color(hex: "1a1535")   // elevated card

    // Brand / accent
    static let cyan     = Color(hex: "00e5ff")
    static let violet   = Color(hex: "c084fc")
    static let green    = Color(hex: "22c55e")
    static let red      = Color(hex: "ef4444")
    static let amber    = Color(hex: "facc15")
    static let blue     = Color(hex: "60a5fa")
    static let indigo   = Color(hex: "818cf8")

    // Text
    static let text     = Color.white
    static let subtext  = Color(hex: "9ca3af")
    static let dim      = Color(hex: "4b5563")

    // Gradients
    static let spaceGradient = LinearGradient(
        colors: [Color(hex: "0a0718"), Color(hex: "160d2e"), Color(hex: "0a0718")],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let cardGradient = LinearGradient(
        colors: [Color(hex: "1a1535"), Color(hex: "110e24")],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )
}

// MARK: - Card modifier
struct CardStyle: ViewModifier {
    var padding: CGFloat = 16

    func body(content: Content) -> some View {
        content
            .padding(padding)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Theme.bg2)
                    .overlay(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .stroke(Theme.dim.opacity(0.4), lineWidth: 0.5)
                    )
            )
    }
}

extension View {
    func cardStyle(padding: CGFloat = 16) -> some View {
        modifier(CardStyle(padding: padding))
    }
}

// MARK: - Star background
struct StarfieldBackground: View {
    let starCount = 120
    let stars: [(x: CGFloat, y: CGFloat, size: CGFloat, opacity: Double)]

    init() {
        var rng = SystemRandomNumberGenerator()
        stars = (0..<starCount).map { _ in
            (
                x: CGFloat.random(in: 0...1, using: &rng),
                y: CGFloat.random(in: 0...1, using: &rng),
                size: CGFloat.random(in: 0.5...2.5, using: &rng),
                opacity: Double.random(in: 0.15...0.9, using: &rng)
            )
        }
    }

    var body: some View {
        GeometryReader { geo in
            ZStack {
                Theme.spaceGradient

                // Subtle nebula blobs
                Circle()
                    .fill(Theme.violet.opacity(0.07))
                    .frame(width: geo.size.width * 0.9)
                    .offset(x: -geo.size.width * 0.15, y: -geo.size.height * 0.05)
                    .blur(radius: 80)

                Circle()
                    .fill(Theme.cyan.opacity(0.05))
                    .frame(width: geo.size.width * 0.7)
                    .offset(x: geo.size.width * 0.3, y: geo.size.height * 0.4)
                    .blur(radius: 70)

                // Stars
                ForEach(stars.indices, id: \.self) { i in
                    let s = stars[i]
                    Circle()
                        .fill(Color.white.opacity(s.opacity))
                        .frame(width: s.size, height: s.size)
                        .position(x: s.x * geo.size.width, y: s.y * geo.size.height)
                }
            }
        }
        .ignoresSafeArea()
    }
}
