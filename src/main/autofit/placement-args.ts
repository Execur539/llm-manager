/**
 * The llama.cpp arguments that decide where a plan's memory goes.
 *
 * Shared by the server launch and by the memory dry run, so what is measured before a load is
 * exactly what the load then asks for. Anything that does not change llama.cpp's own allocations
 * (ports, threads, prompt cache, the vision projector, a draft model) stays with the caller.
 */

import type { FitPlan, ModelArchInfo } from '@shared/types'
import { nglFor } from './engine'

export function placementArgs(
  modelPath: string,
  arch: Pick<ModelArchInfo, 'contextLength'> | null | undefined,
  plan: FitPlan
): string[] {
  const args = [
    '--model', modelPath,
    '--ctx-size', String(plan.contextLength),
    '--n-gpu-layers', String(nglFor(plan)),
    '--batch-size', String(plan.batchSize),
    '--parallel', '1'
  ]
  if (plan.ubatchSize) args.push('--ubatch-size', String(plan.ubatchSize))
  if (plan.kvType !== 'f16') args.push('--cache-type-k', plan.kvType)
  if ((plan.kvTypeV ?? plan.kvType) !== 'f16') args.push('--cache-type-v', plan.kvTypeV ?? plan.kvType)

  const trained = arch?.contextLength ?? 0
  if (trained > 0 && plan.contextLength > trained) {
    args.push(
      '--rope-scaling', 'yarn',
      '--rope-scale', (plan.contextLength / trained).toFixed(5),
      '--yarn-orig-ctx', String(trained)
    )
  }
  if (plan.flashAttention) args.push('--flash-attn', 'on')

  // Whole blocks per card where the plan has them, so the split lands exactly where it was costed.
  const blocksPerCard = plan.layerSplit ?? []
  if (blocksPerCard.length > 1) {
    args.push('--tensor-split', blocksPerCard.join(','))
  } else if (plan.tensorSplit.length > 1) {
    args.push('--tensor-split', plan.tensorSplit.map((s) => s.toFixed(3)).join(','))
  }

  const cpuMoe = Math.max(0, Math.floor(Number(plan.cpuMoeLayers) || 0))
  if (cpuMoe > 0) args.push('--n-cpu-moe', String(cpuMoe))
  if (typeof plan.overrideTensors === 'string' && plan.overrideTensors.trim()) {
    args.push('--override-tensor', plan.overrideTensors.trim())
  }
  return args
}

/** The rope-scaling warning the launch logs, kept beside the arguments it describes. */
export function ropeScale(arch: Pick<ModelArchInfo, 'contextLength'> | null | undefined, context: number): number {
  const trained = arch?.contextLength ?? 0
  return trained > 0 && context > trained ? context / trained : 1
}
