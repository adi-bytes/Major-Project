import { Layout } from "@/components/Layout";
import { useAppStore } from "@/store/appStore";
import { OverviewView } from "@/views/OverviewView";
import { ForwardPredictionView } from "@/views/ForwardPredictionView";
import { ReverseModeView } from "@/views/ReverseModeView";
import { ValidationDashboardView } from "@/views/ValidationDashboardView";
import { ModelDetailsView } from "@/views/ModelDetailsView";
import { DesignView } from "@/views/DesignView";

function AppContent() {
  const { activeTab } = useAppStore();

  return (
    <Layout>
      {activeTab === "overview" && <OverviewView />}
      {activeTab === "forward" && <ForwardPredictionView />}
      {activeTab === "reverse" && <ReverseModeView />}
      {activeTab === "validation" && <ValidationDashboardView />}
      {activeTab === "model" && <ModelDetailsView />}
      {activeTab === "design" && <DesignView />}
    </Layout>
  );
}

export default AppContent;
