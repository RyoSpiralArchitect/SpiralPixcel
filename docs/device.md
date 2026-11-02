# Pixelをつないで再開する

MacにAndroid SDK Platform-Toolsを入れ、開発者向けオプションでUSBデバッグを有効にする。データ対応ケーブルで接続し、Pixelに表示されるMacの鍵を確認する。以下はUSBで接続した1台を対象にする。

    adb -d get-state
    adb -d shell getprop ro.product.model
    adb -d shell vm list

開発者向けオプションのLinux開発環境を有効にし、標準「ターミナル」でDebianを起動する。AndroidのDownloadはVMの /mnt/shared/Download から読める。ビルドと計算はVMホームで行い、共有領域は受け渡しに使う。

## 作業中の画面

元の値を控えてから、ACまたはUSBでの給電中に画面を維持する設定を入れる。

    adb -d shell settings get global stay_on_while_plugged_in
    adb -d shell settings put global stay_on_while_plugged_in 3
    adb -d shell dumpsys power | grep -E 'mStayOn=|mPlugType=|mMaximumScreenOffTimeoutFromDeviceAdmin'

検証端末の元の値は0。戻すには最後の値を0にして settings put を再実行する。端末管理者による上限が適用されると mStayOn=false のままになる。画面がロックされたら端末で解除する。

## USBとSSH

Macは信頼済みのUSB ADBからAndroidの nc を起動し、VMのプライベートアドレスのSSHに接続する。Terminalの外部ポート転送設定は不要。接続開始時はVMのホスト鍵を照合し、専用のEd25519鍵で droid として認証する。秘密鍵と既知のホストはMacの `~/Library/Application Support/SpiralPixcel/` に保存する。 Linux等では `SPIRALPIXCEL_STATE` でこの場所を変更できる。

再起動後は標準ターミナルを開く。通常はVM用ネットワークの近隣アドレスから再発見する。到達できない場合はDebianの `hostname -I` を確認し、Macで `SPIRALPIXCEL_VM_IP=<VMのIPv4> ./pixel-lab doctor` を試す。VMを作り直した場合は再度 pair と trust を行う。鍵の登録を取り消すには、Debianの `~/.ssh/authorized_keys` から対応する末尾 SpiralPixcel の行を取り除く。

既存のQwen実験はAndroidの Download/SpiralPixel 配下にある。この公開ツールは別の Download/SpiralPixcel を鍵登録時の受け渡しに使う。計算時のアップロードと回収はSSH経由で行う。

## この実機の出発点

Pixel 11 Pro / Android 17の標準ターミナルで、Debianが公開したVulkan APIは1.3.0、デバイス名はPowerVR C-Series CXTP-48-1536 MC1。Androidネイティブの測定とは別に記録する。

先行するQwen2.5-0.5B-Instruct Q4_K_Mの実験では、llama.cpp b11960のVulkanバックエンドが起動時に VK_KHR_16bit_storage を必須名として要求した。ゲストは対応するコア機能を公開し、拡張名を列挙しなかったため、コア機能を検査して名前があるときだけ有効にする変更を使った。25/25層の短い生成を確認した設定は ctx=512、batch=16、ubatch=16、threads=4、Flash Attention off、warmup off。今回の整数・拡散シェーダーは32ビットのみを使用し、モデルやこのパッチの配備は不要。

- [Android 17 標準Terminalのソース側ドキュメント](https://android.googlesource.com/platform/packages/modules/Virtualization/+/refs/tags/android-17.0.0_r1/docs/custom_vm.md)
- [16ビットストレージとVulkan 1.1への昇格](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_16bit_storage.html)
- [llama.cpp b11960](https://github.com/ggml-org/llama.cpp/releases/tag/b11960)
