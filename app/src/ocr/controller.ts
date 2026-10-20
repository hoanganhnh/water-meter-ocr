import type {Spec} from '../specs/NativeMeterOcr';

export type DiagnosticState = {status: 'idle' | 'busy' | 'done' | 'error'; result?: string; error?: string};

export class DiagnosticController {
  private generation = 0;
  private busy = false;

  invalidate() {
    this.generation += 1;
  }

  async run(uri: string, module: Spec, onState: (state: DiagnosticState) => void, parity = false): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    const request = ++this.generation;
    onState({status: 'busy'});
    try {
      const result = parity ? await module.runImageParity(uri) : await module.runImageDiagnostic(uri);
      if (request === this.generation) onState({status: 'done', result});
    } catch (error) {
      if (request === this.generation) onState({status: 'error', error: error instanceof Error ? error.message : String(error)});
    } finally {
      this.busy = false;
      if (request !== this.generation) onState({status: 'idle'});
    }
  }
}
