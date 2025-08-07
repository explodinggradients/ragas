import { AppSidebar } from "@/components/app-sidebar"
import {
  SidebarInset,
  SidebarProvider,
} from "@/components/ui/sidebar"
import { CommonLayout } from "@/components/CommonLayout"
import { BreadcrumbProvider } from "@/components/Breadcrumb/provider"
import {Router} from "@/Router";

export default function App() {
  return (
    <SidebarProvider>
      <BreadcrumbProvider>
        <AppSidebar />
        <SidebarInset>
          <CommonLayout>
            <Router />
          </CommonLayout>
        </SidebarInset>
      </BreadcrumbProvider>
    </SidebarProvider>
  )
}
