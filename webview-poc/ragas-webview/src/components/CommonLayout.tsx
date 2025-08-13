import type { ReactNode } from 'react';
import { Separator } from '@/components/ui/separator';
import { SidebarTrigger } from '@/components/ui/sidebar';
import { BreadcrumbNav } from './Breadcrumb';

type CommonLayoutProps = {
  children: ReactNode;
};

export function CommonLayout({ children }: CommonLayoutProps) {
  return (
    <>
      <header className="flex h-16 shrink-0 items-center gap-2 border-b">
        <div className="flex items-center gap-2 px-3">
          <SidebarTrigger />
          <Separator orientation="vertical" className="mr-2 h-4" />
          <BreadcrumbNav />
        </div>
      </header>
      <div className="flex flex-1 flex-col gap-4 p-4">
        {children}
      </div>
    </>
  );
}
