# SpiralPixcel 🌀

SpiralRealityのPixel実験ノートと、標準ターミナルのDebian VMでVulkan computeを動かすための小さな道具箱。

[端末設定と再開手順](docs/device.md) · [MITライセンス](LICENSE)

## Macから実行

MacにPython 3.9以降・OpenSSH・Android Platform-Tools、Pixelの標準ターミナルにDebianとGPUアクセラレーションが必要。Debianで次を一度実行する。

    sudo apt-get update
    sudo apt-get install --no-install-recommends build-essential libvulkan-dev glslc openssh-server
    sudo systemctl start ssh

Macでこのリポジトリを開き、USB接続中に専用鍵を登録する。

    ./pixel-lab pair

表示されたコマンドをPixelのDebianで実行し、画面に出たホスト鍵の指紋をMacの `./pixel-lab trust SHA256:…` に渡す。次回からは、USBをつないでDebianを起動すれば続けられる。

    ./pixel-lab doctor

37×7個の32ビット整数をGPUで2倍し、259個すべてを期待値と照合する。GPU名、API、公開拡張、キュー、時間、入力・出力・シェーダーのハッシュを記録する。CPU型Vulkanデバイスでの実行は実機用コマンドでは失敗として扱う。CIは明示的にソフトウェアVulkanを許可する。

各実行の結果はMacのカレントディレクトリの `runs/<ID>/`、ソースと結果の控えはVMの `~/.local/share/SpiralPixcel/runs/<ID>/` に残る。回収だけやり直すには `./pixel-lab collect <ID>`。失敗時は `build-run.log` を見る。
