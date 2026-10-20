import React, {useEffect, useRef, useState} from 'react';
import {AppState, Image, Platform, Pressable, ScrollView, StatusBar, StyleSheet, Text, View} from 'react-native';
import ImagePicker, {type Image as PickedImage} from 'react-native-image-crop-picker';
import {SafeAreaView} from 'react-native-safe-area-context';
import {DiagnosticController, type DiagnosticState} from '../ocr/controller';
import {getNativeOcr} from '../ocr/native';
import {readingView} from '../ocr/result';

function cleanupCrop(path: string) {
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

  const chooseAndRead = async () => {
    if (mediaActive.current || nativeTask.current) return;
    mediaActive.current = true;
    setPickerBusy(true);
    controller.invalidate();
    setState({status: 'idle'});
    let selectedPath: string | undefined;
    try {
      const selected = await ImagePicker.openPicker({
        mediaType: 'photo',
        cropping: false,
        includeExif: true,
      });
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
      const message = error instanceof Error ? error.message : String(error);
      if (selectedPath) cleanupCrop(selectedPath);
      if (mounted.current && !message.toLowerCase().includes('cancel')) setState({status: 'error', error: `Không mở được ảnh: ${message}`});
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

  const runParityDiagnostic = async () => {
    if (!photo?.path || nativeTask.current || mediaActive.current) return;
    try {
      const task = controller.run(photo.path, getNativeOcr(), updateOcrState, true);
      nativeTask.current = task;
      await task;
    } catch (error) {
      updateOcrState({status: 'error', error: error instanceof Error ? error.message : String(error)});
    } finally {
      nativeTask.current = null;
    }
  };

  const chooseExactCropForParity = async () => {
    if (mediaActive.current || nativeTask.current) return;
    mediaActive.current = true;
    setPickerBusy(true);
    controller.invalidate();
    let selectedPath: string | undefined;
    try {
      const selected = await ImagePicker.openPicker({mediaType: 'photo', cropping: false, includeExif: true});
      selectedPath = selected.path;
      if (!mounted.current) { cleanupCrop(selected.path); return; }
      await getNativeOcr().validateImage(selected.path);
      if (!mounted.current) { cleanupCrop(selected.path); return; }
      if (photo?.path) cleanupCrop(photo.path);
      photoRef.current = selected;
      setPhoto(selected);
      setState({status: 'idle'});
    } catch (error) {
      if (selectedPath) cleanupCrop(selectedPath);
      const message = error instanceof Error ? error.message : String(error);
      if (mounted.current && !message.toLowerCase().includes('cancel')) setState({status: 'error', error: `Parity fixture: ${message}`});
    } finally {
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
        <Text style={styles.body}>Chọn ảnh. Ứng dụng hoàn toàn chạy offline.</Text>
        <Pressable accessibilityRole="button" onPress={chooseAndRead} disabled={pickerBusy || state.status === 'busy' || Platform.OS !== 'android'} style={({pressed}) => [styles.button, pressed && styles.pressed, Platform.OS !== 'android' && styles.disabled]}>
          <Text style={styles.buttonText}>{Platform.OS !== 'android' ? 'Android hiện tại · iPhone ở phase 6' : pickerBusy ? 'Đang mở thư viện…' : state.status === 'busy' ? 'Đang tìm và đọc vùng số…' : photo ? 'Chọn ảnh khác' : 'Chọn ảnh đồng hồ'}</Text>
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
          <Pressable accessibilityRole="button" onPress={() => processImage(photo.path)} disabled={pickerBusy || state.status === 'busy'} style={[styles.button, styles.secondary, (pickerBusy || state.status === 'busy') && styles.disabled]}>
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
