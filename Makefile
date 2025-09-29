clippy: fmt
	CANDLE_FLASH_ATTN_BUILD_DIR=~/.cache/candle_flash_attn \
    ORT_DYLIB_PATH=/usr/local/onnxruntime/lib/libonnxruntime.so \
	cargo clippy \
    		--fix \
    		--allow-dirty \
    		--all-targets \
    		--all-features \
    		-- -D warnings

fmt:
	CANDLE_FLASH_ATTN_BUILD_DIR=~/.cache/candle_flash_attn \
    ORT_DYLIB_PATH=/usr/local/onnxruntime/lib/libonnxruntime.so \
    cargo fmt

test:
	CANDLE_FLASH_ATTN_BUILD_DIR=~/.cache/candle_flash_attn \
        ORT_DYLIB_PATH=/usr/local/onnxruntime/lib/libonnxruntime.so \
        cargo test \
        	--all-targets \
         	--workspace
