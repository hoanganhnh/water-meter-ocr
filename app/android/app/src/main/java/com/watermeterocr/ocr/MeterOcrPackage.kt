package com.watermeterocr.ocr

import com.facebook.react.BaseReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.module.model.ReactModuleInfo
import com.facebook.react.module.model.ReactModuleInfoProvider
import com.facebook.react.turbomodule.core.interfaces.TurboModule
import android.util.Log

class MeterOcrPackage : BaseReactPackage() {
  override fun getModule(name: String, reactContext: ReactApplicationContext): NativeModule? {
    if (name != MeterOcrModule.NAME) return null
    val module = MeterOcrModule(reactContext)
    Log.i("MeterOcrPackage", "getModule($name): turbo=${module is TurboModule}, actualName=${module.name}")
    return module
  }

  override fun getReactModuleInfoProvider() = ReactModuleInfoProvider {
    Log.i("MeterOcrPackage", "providing TurboModule metadata")
    mapOf(
      MeterOcrModule.NAME to ReactModuleInfo(
        MeterOcrModule.NAME,
        MeterOcrModule::class.java.name,
        false,
        false,
        false,
        ReactModuleInfo.classIsTurboModule(MeterOcrModule::class.java),
      ),
    )
  }
}
