import React, {useEffect, useRef, useState} from 'react';
import {AppState, Image, Platform, Pressable, ScrollView, StatusBar, StyleSheet, Text, View} from 'react-native';
import ImagePicker, {type Image as PickedImage} from 'react-native-image-crop-picker';
import {SafeAreaView} from 'react-native-safe-area-context';
import {DiagnosticController, type DiagnosticState} from '../ocr/controller';
import {getNativeOcr} from '../ocr/native';
import {readingView} from '../ocr/result';

function cleanupCrop(path: string) {
  if (path.includes('/cache/meter-realtime-')) {
    // Private cache needs no storage permission, including on Android 7–10.
    try { getNativeOcr().releaseRealtimeImage(path).catch(() => undefined); }
    catch { /* The React context may already have been disposed. */ }
    return;
  }
  ImagePicker.cleanSingle(path).catch(() => undefined);
}

export default function MeterScreen() {
  const [photo, setPhoto] = useState<PickedImage | null>(null);
  const [state, setState] = useState<DiagnosticState>({status: 'idle'});
  const [pickerBusy, setPickerBusy] = useState(false);
  const controller = useRef(new DiagnosticController()).current;
  const mounted = useRef(true);
  const photoRef = useRef<PickedImage | null>(null);
  const nativeTask = useRef<Promise<void> | null>(null);
  const mediaActive = useRef(false);
  const automaticRead = useRef(false);
  const [previewWidth, setPreviewWidth] = useState(0);
  const reading = state.result ? readingView(state.result) : null;
  useEffect(() => { photoRef.current = photo; }, [photo]);
  const updateOcrState = (next: DiagnosticState) => { if (mounted.current) setState(next); };

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      controller.invalidate();
      const path = photoRef.current?.path;
      if (path) {
        if (nativeTask.current) nativeTask.current.finally(() => cleanupCrop(path));
        else cleanupCrop(path);
      }
    };
  }, [controller]);
  useEffect(() => {
    const subscription = AppState.addEventListener('change', next => {
      // Android can deliver the picker's pause after its result callback. The
      // automatic job still owns this selected photo; let that bounded job
      // settle. Unmount/new-selection guards continue to invalidate stale work.
      if (next !== 'active' && !automaticRead.current) {
        controller.invalidate();
      }
    });
    return () => subscription.remove();
  }, [controller]);

  const chooseAndRead = async (source: 'gallery' | 'camera') => {
    if (mediaActive.current || nativeTask.current) return;
    mediaActive.current = true;
    setPickerBusy(true);
    controller.invalidate();
    setState({status: 'idle'});
    let selectedPath: string | undefined;
    try {
      const options = {
        mediaType: 'photo',
        cropping: false,
        includeExif: true,
      } as const;
      const selected = source === 'camera'
        ? await ImagePicker.openCamera({
          ...options,
          useFrontCamera: false,
          compressImageMaxWidth: 2560,
          compressImageMaxHeight: 2560,
          compressImageQuality: 0.95,
        })
        : await ImagePicker.openPicker(options);
      selectedPath = selected.path;
      if (!mounted.current) { cleanupCrop(selected.path); return; }
      if (selected.size > 25 * 1024 * 1024) throw new Error('Ảnh vượt giới hạn 25 MB; hãy chọn tệp nhỏ hơn.');
      await getNativeOcr().validateImage(selected.path);
      if (!mounted.current) { cleanupCrop(selected.path); return; }
      if (photoRef.current?.path && photoRef.current.path !== selected.path) cleanupCrop(photoRef.current.path);
      photoRef.current = selected;
      setPhoto(selected);
      selectedPath = undefined; // Ownership transferred to the displayed photo.
      mediaActive.current = false;
      setPickerBusy(false);
      automaticRead.current = true;
      try { await processImage(selected.path); }
      finally { automaticRead.current = false; }
    } catch (error) {
      const failure = error as {code?: string; message?: string} | null;
      const code = failure?.code;
      const message = failure?.message ?? String(error);
      if (selectedPath) cleanupCrop(selectedPath);
      if (mounted.current && code !== 'E_PICKER_CANCELLED' && !message.toLowerCase().includes('cancel')) {
        const description = code === 'E_NO_CAMERA_PERMISSION'
          ? 'Chưa có quyền camera. Vào Cài đặt → Ứng dụng → WaterMeterOCR → Quyền để cho phép camera, rồi thử lại.'
          : code === 'E_CAMERA_IS_NOT_AVAILABLE' || code === 'E_CANNOT_LAUNCH_CAMERA' || code === 'E_PICKER_CANNOT_RUN_CAMERA_ON_SIMULATOR'
            ? 'Không có camera khả dụng. Bạn vẫn có thể chọn ảnh từ thư viện.'
            : `Không mở được ảnh: ${message}`;
        setState({status: 'error', error: description});
      }
    } finally {
      mediaActive.current = false;
      if (mounted.current) setPickerBusy(false);
    }
  };

  const processImage = async (path: string) => {
    if (nativeTask.current || mediaActive.current) return;
    try {
      const task = controller.run(path, getNativeOcr(), updateOcrState);
      nativeTask.current = task;
      await task;
    } catch (error) {
      updateOcrState({status: 'error', error: error instanceof Error ? error.message : String(error)});
    } finally {
      nativeTask.current = null;
    }
  };

  const scanRealtime = async () => {
    if (mediaActive.current || nativeTask.current) return;
    mediaActive.current = true;
    setPickerBusy(true);
    controller.invalidate();
    let ownedPath: string | undefined;
    try {
      const response = JSON.parse(await getNativeOcr().startRealtime()) as {
        cancelled?: boolean; uri: string; width: number; height: number; size: number; result: object;
      };
      if (response.cancelled) return;
      ownedPath = response.uri;
      if (!mounted.current) return;
      if (photoRef.current?.path) cleanupCrop(photoRef.current.path);
      const selected: PickedImage = {path: response.uri, width: response.width,
        height: response.height, size: response.size, mime: 'image/jpeg'};
      photoRef.current = selected;
      setPhoto(selected);
      ownedPath = undefined;
      setState({status: 'done', result: JSON.stringify(response.result)});
    } catch (error) {
      updateOcrState({status: 'error', error: error instanceof Error ? error.message : String(error)});
    } finally {
      if (ownedPath) cleanupCrop(ownedPath);
      mediaActive.current = false;
      if (mounted.current) setPickerBusy(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" />
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.kicker}>WATER METER</Text>
        <Text style={styles.title}>Đọc ảnh</Text>
        <Text style={styles.body}>Chụp ảnh, chọn ảnh hoặc quét realtime. Ứng dụng hoàn toàn chạy offline.</Text>
        <Pressable testID="choose-photo" accessibilityRole="button" onPress={() => chooseAndRead('gallery')} disabled={pickerBusy || state.status === 'busy' || Platform.OS !== 'android'} style={({pressed}) => [styles.button, pressed && styles.pressed, (pickerBusy || state.status === 'busy' || Platform.OS !== 'android') && styles.disabled]}>
          <Text style={styles.buttonText}>{Platform.OS !== 'android' ? 'Android hiện tại · iPhone ở phase 6' : pickerBusy ? 'Đang mở ảnh…' : state.status === 'busy' ? 'Đang tìm và đọc vùng số…' : photo ? 'Chọn ảnh khác' : 'Chọn ảnh từ thư viện'}</Text>
        </Pressable>
        <Pressable testID="take-photo" accessibilityRole="button" onPress={() => chooseAndRead('camera')} disabled={pickerBusy || state.status === 'busy' || Platform.OS !== 'android'} style={({pressed}) => [styles.button, styles.cameraButton, pressed && styles.pressed, (pickerBusy || state.status === 'busy' || Platform.OS !== 'android') && styles.disabled]}>
          <Text style={styles.buttonText}>Chụp ảnh đồng hồ</Text>
        </Pressable>
        <Pressable testID="scan-realtime" accessibilityRole="button" onPress={scanRealtime} disabled={pickerBusy || state.status === 'busy' || Platform.OS !== 'android'} style={({pressed}) => [styles.button, pressed && styles.pressed, (pickerBusy || state.status === 'busy' || Platform.OS !== 'android') && styles.disabled]}>
          <Text style={styles.buttonText}>Nhận diện realtime</Text>
        </Pressable>
        {photo ? <View style={styles.card}>
          <View style={styles.photo} onLayout={event => setPreviewWidth(event.nativeEvent.layout.width)}>
            <Image source={{uri: photo.path}} style={StyleSheet.absoluteFill} resizeMode="contain" />
            {reading?.region && previewWidth > 0 ? (() => {
              const {box, size} = reading.region;
              const scale = Math.min(previewWidth / size[0], 260 / size[1]);
              return <View testID="detected-region" pointerEvents="none" style={[styles.region, {left: (previewWidth - size[0] * scale) / 2 + box[0] * scale, top: (260 - size[1] * scale) / 2 + box[1] * scale, width: (box[2] - box[0]) * scale, height: (box[3] - box[1]) * scale}]} />;
            })() : null}
          </View>
          <Pressable testID="retry-reading" accessibilityRole="button" onPress={() => processImage(photo.path)} disabled={pickerBusy || state.status === 'busy'} style={[styles.button, styles.secondary, (pickerBusy || state.status === 'busy') && styles.disabled]}>
            <Text style={styles.buttonText}>{state.status === 'busy' ? 'Đang đọc trên thiết bị…' : 'Thử đọc lại'}</Text>
          </Pressable>
        </View> : null}
        {state.status === 'done' && reading ? <View style={styles.result}>
          <Text style={styles.resultTitle}>{reading.digits ? 'Chỉ số thử nghiệm' : 'Cần kiểm tra lại'}</Text>
          {reading.digits ? <Text selectable testID="meter-reading" style={styles.reading}>{reading.digits}</Text> : null}
          <Text>{reading.message}</Text>
          {reading.elapsedMs !== null ? <Text style={styles.meta}>{Math.round(reading.elapsedMs)} ms · xử lý offline</Text> : null}
        </View> : null}
        {state.status === 'error' ? <View style={styles.error}><Text style={styles.resultTitle}>Chưa xử lý được</Text><Text>{state.error}</Text></View> : null}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {flex: 1, backgroundColor: '#f3f6f2'},
  content: {padding: 22, gap: 16},
  kicker: {color: '#36775c', fontSize: 12, fontWeight: '800', letterSpacing: 1.3},
  title: {color: '#14271e', fontSize: 30, fontWeight: '800'},
  body: {color: '#52665a', fontSize: 16, lineHeight: 23},
  button: {alignItems: 'center', backgroundColor: '#176b52', borderRadius: 14, minHeight: 52, justifyContent: 'center', padding: 14},
  secondary: {backgroundColor: '#285b49', marginTop: 12},
  cameraButton: {backgroundColor: '#285b49'},
  pressed: {opacity: 0.75},
  disabled: {opacity: 0.6},
  buttonText: {color: '#fff', fontSize: 15, fontWeight: '700'},
  card: {backgroundColor: '#fff', borderRadius: 18, padding: 14},
  photo: {width: '100%', height: 260, backgroundColor: '#e9eee9', borderRadius: 10},
  region: {position: 'absolute', borderWidth: 2, borderColor: '#12a673'},
  meta: {color: '#738177', fontSize: 12, marginTop: 8},
  notice: {backgroundColor: '#fff1d2', borderRadius: 14, padding: 16, gap: 6},
  noticeTitle: {color: '#75520d', fontSize: 15, fontWeight: '800'},
  noticeText: {color: '#66562f', lineHeight: 21},
  result: {backgroundColor: '#e6eee8', borderRadius: 14, padding: 16, gap: 8},
  resultTitle: {fontWeight: '800', color: '#233e30'},
  mono: {fontFamily: 'monospace', color: '#1b3025'},
  reading: {fontFamily: 'monospace', color: '#14271e', fontSize: 40, fontWeight: '800', letterSpacing: 3},
  error: {backgroundColor: '#fde9e6', borderRadius: 14, padding: 16, gap: 8},
});
