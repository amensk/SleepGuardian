import SwiftUI

struct ContentView: View {
    @StateObject private var appState = AppState.shared
    @State private var selectedTab: Tab = .status

    enum Tab: String, CaseIterable {
        case status  = "Status"
        case alerts  = "Alerts"
        case trends  = "Trends"
        case reports = "Reports"

        var icon: String {
            switch self {
            case .status:  return "shield.lefthalf.filled"
            case .alerts:  return "bell.fill"
            case .trends:  return "waveform.path.ecg"
            case .reports: return "chart.bar.fill"
            }
        }
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            StarfieldBackground()

            // Content
            Group {
                switch selectedTab {
                case .status:  StatusView()
                case .alerts:  AlertsView()
                case .trends:  TrendsView()
                case .reports: ReportsView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Custom Tab Bar
            CustomTabBar(selectedTab: $selectedTab)
        }
        .environmentObject(appState)
        .ignoresSafeArea(edges: .bottom)
    }
}

// MARK: - Custom Tab Bar
struct CustomTabBar: View {
    @Binding var selectedTab: ContentView.Tab

    var body: some View {
        HStack(spacing: 0) {
            ForEach(ContentView.Tab.allCases, id: \.self) { tab in
                TabBarItem(tab: tab, isSelected: selectedTab == tab)
                    .onTapGesture {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                            selectedTab = tab
                        }
                    }
            }
        }
        .padding(.horizontal, 8)
        .padding(.top, 12)
        .padding(.bottom, 28)
        .background(
            Rectangle()
                .fill(Color(hex: "0d0b1e").opacity(0.95))
                .overlay(
                    Rectangle()
                        .frame(height: 0.5)
                        .foregroundColor(Color(hex: "2d2550"))
                        .frame(maxHeight: .infinity, alignment: .top)
                )
        )
    }
}

struct TabBarItem: View {
    let tab: ContentView.Tab
    let isSelected: Bool

    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: tab.icon)
                .font(.system(size: 22, weight: isSelected ? .semibold : .regular))
                .foregroundColor(isSelected ? Theme.cyan : Theme.subtext)

            Text(tab.rawValue)
                .font(.system(size: 10, weight: .medium))
                .foregroundColor(isSelected ? Theme.cyan : Theme.subtext)
        }
        .frame(maxWidth: .infinity)
        .contentShape(Rectangle())
    }
}
