import { Routes, Route, Navigate } from 'react-router-dom'
import {Datasets} from "@/containers/Datasets/Datasets.tsx";
import {COMMON_ROUTES} from "@/Router/routes.ts";
import {Experiments} from "@/containers/Experiments/Experiments.tsx";

export function Router() {
  return (
    <Routes>
      <Route path={COMMON_ROUTES.root} element={<Navigate to={COMMON_ROUTES.datasets} replace />} />
      <Route path={COMMON_ROUTES.datasets} element={<Datasets />} />
      <Route path={COMMON_ROUTES.experiments} element={<Experiments />} />
      <Route path="*" element={<Navigate to={COMMON_ROUTES.datasets} replace />} />
    </Routes>
  )
}
