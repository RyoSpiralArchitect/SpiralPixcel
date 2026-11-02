#!/bin/sh
# Installed from a temporary, hash-verified copy containing the Mac public key.
set -eu
umask 077
mkdir -p "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys"
chmod 700 "$HOME/.ssh"
chmod 600 "$HOME/.ssh/authorized_keys"
entry="restrict $mac_public_key"
if ! grep -Fqx "$entry" "$HOME/.ssh/authorized_keys"; then
    printf '\n%s\n' "$entry" >> "$HOME/.ssh/authorized_keys"
fi
out=/mnt/shared/Download/SpiralPixcel
cp /etc/ssh/ssh_host_ed25519_key.pub "$out/vm-host.pub"
hostname -I | awk '{print $1}' > "$out/vm-ip"
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
