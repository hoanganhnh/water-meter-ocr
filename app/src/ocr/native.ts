import {NativeModules, Platform, TurboModuleRegistry} from 'react-native';
import type {Spec} from '../specs/NativeMeterOcr';

export function getNativeOcr(): Spec {
  if (Platform.OS !== 'android') {
    throw new Error('Ảnh local hiện chỉ được hỗ trợ trên Android; iPhone thuộc phase 6.');
  }
  const proxy = (globalThis as typeof globalThis & {
    __turboModuleProxy?: (name: string) => Spec | null;
    nativeModuleProxy?: Record<string, Spec | undefined>;
  });
  const native = TurboModuleRegistry.get<Spec>('NativeMeterOcr')
    ?? proxy.__turboModuleProxy?.('NativeMeterOcr')
    ?? proxy.nativeModuleProxy?.NativeMeterOcr
    ?? (NativeModules.NativeMeterOcr as Spec | undefined);
  if (!native) throw new Error('Không tìm thấy NativeMeterOcr TurboModule trong Android runtime.');
  return native;
}
