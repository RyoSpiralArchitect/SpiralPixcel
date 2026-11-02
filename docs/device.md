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
