import type {TurboModule} from 'react-native';
import {TurboModuleRegistry} from 'react-native';

export interface Spec extends TurboModule {
  runRuntimeSmoke(): Promise<string>;
  validateImage(uri: string): Promise<string>;
  runImageDiagnostic(uri: string): Promise<string>;
  runImageParity(uri: string): Promise<string>;
}

// iOS is scaffold-only until Phase 6 registers the native implementation.
export default TurboModuleRegistry.get<Spec>('NativeMeterOcr');
