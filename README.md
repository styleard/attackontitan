# Titan Giants (AOT tarzı mod) — Forge 1.17.1

Orijinal tasarımlı iki dev: **Akılsız Dev** (6 blok, düşman) ve **Savaş Devi** (15 blok, oyuncu dönüşümü).

## GitHub ile .jar üretme
1. github.com → **New repository** (ör. `aotmod117`).
2. Bu klasörün **içindeki her şeyi** (`.github` klasörü dahil) repoya yükle (Add file → Upload files).
   `.github` yüklenmezse: Add file → Create new file → ad kısmına `.github/workflows/build.yml` yaz, içeriğini projedeki dosyadan yapıştır.
3. **Actions** sekmesi → *Build mod* çalışır (çalışmadıysa *Run workflow*). Yaklaşık 5-8 dakika sürer.
4. Bitince çalışmanın sayfasında **Artifacts → aotmod-1.17.1-jar** indir → zip içinden `aotmod-1.17.1-1.0.0.jar`.

## Oyunda kullanım
- Gerekli: Minecraft **1.17.1**, **Forge 37.1.1**, **GeckoLib 3.0.x (Forge 1.17.1)**. Jar'ı ve GeckoLib'i `mods` klasörüne koy.
- Derlemede GeckoLib sürümü `gradle.properties` içindeki `geckolib_version=3.0.+` ile seçilir. Sürüm bulunamazsa
  GeckoLib'in sürüm listesinden gerçek bir 1.17.1 sürümünü yazıp tekrar çalıştır.
- **Savaş Devi Serumu** tarifi (şekilsiz): Cam şişe + Elmas + Ghast gözyaşı + Altın elma. Yaratıcı modda "titanmod" sekmesinde de var.
- Serumu içince efekt listesinde **Savaş Devi Gücü** görünür.
- **Shift + M**: Savaş Devine dönüş (şimşek, patlama, kükreme, ağız açılır). Tekrar Shift + M: eski hale dönüş.
- Dönüşmek için etrafında 4x15 blokluk boş alan gerekir.
- Akılsız Dev: yaratıcı modda yumurtasıyla ya da `/summon titanmod:mindless_giant`; overworld içinde nadiren de doğar.

## Assetleri yeniden üretmek
`python3 tools/gen_assets.py` (pillow + numpy). Model, doku ve animasyonlar `src/main/resources/assets/titanmod/` altına yazılır.
