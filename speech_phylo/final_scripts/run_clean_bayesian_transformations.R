#!/usr/bin/env Rscript

# export_results_to_json.R
#
# Usage:
#   Rscript speech_phylo/export_results_to_json.R

# ─── 0. Setup ──────────────────────────────────────────────────────
suppressPackageStartupMessages({
    library(ape)
    library(brms)
    library(dplyr)
    library(tibble)
    library(jsonlite)
})

# ensure output dir exists
out_dir <- "speech_phylo/final_phyloregression_results"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ─── 1. Load BEAST helper & data ─────────────────────────────────
source("speech_phylo/beast.R")  # must define read.annot.beast() & extract_beast_metadata()

tree_names <- c("heggarty2024", "long_v3_CCD0.0.25")

coordinate_shift <- 10

apply_coord_scaling <- function(vec, method, shift = coordinate_shift) {
    forward_transform <- NULL
    inverse_pre_transform <- NULL
    description <- NULL
    shift_value <- NA_real_

    if (method == "standard") {
        forward_transform <- function(x) x
        inverse_pre_transform <- function(t) t
        description <- "scale(x)"
    } else if (method == "sqrt_plus_10") {
        if (any(vec + shift <= 0)) {
            stop(sprintf("sqrt_plus_10 requires all values to be greater than -%s", shift))
        }
        forward_transform <- function(x) sqrt(x + shift)
        inverse_pre_transform <- function(t) (t^2) - shift
        description <- sprintf("scale(sqrt(x + %s))", shift)
        shift_value <- shift
    } else if (method == "log_plus_10") {
        if (any(vec + shift <= 0)) {
            stop(sprintf("log_plus_10 requires all values to be greater than -%s", shift))
        }
        forward_transform <- function(x) log(x + shift)
        inverse_pre_transform <- function(t) exp(t) - shift
        description <- sprintf("scale(log(x + %s))", shift)
        shift_value <- shift
    } else {
        stop(sprintf("Unknown coordinate scaling method: %s", method))
    }

    transformed <- forward_transform(vec)
    scaled <- scale(transformed)
    center <- as.numeric(attr(scaled, "scaled:center"))
    scale_val <- as.numeric(attr(scaled, "scaled:scale"))

    inverse_transform <- function(z) {
        if (is.na(scale_val) || is.na(z)) {
            return(NA_real_)
        }
        inverse_pre_transform(center + z * scale_val)
    }

    original_center <- inverse_pre_transform(center)
    delta_original_per_sd <- NA_real_
    if (!is.na(scale_val) && scale_val != 0) {
        delta_original_per_sd <- inverse_transform(1) - original_center
        if (isTRUE(all.equal(delta_original_per_sd, 0))) {
            delta_original_per_sd <- NA_real_
        }
    }

    return(list(
        values = as.numeric(scaled),
        center = center,
        scale = scale_val,
        description = description,
        shift = shift_value,
        original_center = original_center,
        delta_original_per_sd = delta_original_per_sd
    ))
}

for (tree_name in tree_names) {

    tree_path     <- paste0("speech_phylo/", tree_name, ".nex")  # BEAST .trees or .nexus file
    glotto_path   <- "speech_phylo/glottolog_df.csv"

    tr            <- read.annot.beast(tree_path)
    glottolog_df  <- read.csv(glotto_path, stringsAsFactors = FALSE)

    # ─── 2. Extract & merge metadata ─────────────────────────────────
    metadata_df <- extract_beast_metadata(tree = tr)
    df <- inner_join(metadata_df, glottolog_df, by = "language") %>%
        column_to_rownames(var = "language")

    # ─── 3. Prune tree & compute cophenetic distances ────────────────
    to_drop <- setdiff(tr$tip.label, rownames(df))
    tr      <- drop.tip(tr, to_drop)
    coph_df <- as.data.frame(cophenetic(unroot(tr)))

    # 4) compute raw phylogenetic covariance (shared branch lengths)
    V_raw <- vcv(unroot(tr))   # n×n matrix

    # 5) write metadata.csv (one row per tip, with language column)
    meta_out_df <- df %>% rownames_to_column("language")
    write.csv(
        meta_out_df,
        file = file.path(out_dir, paste0("metadata_", tree_name, ".csv")),
        row.names = FALSE,
        quote = TRUE
    )

    # 6) write V_raw.csv (with row & col names)
    write.csv(
        V_raw,
        file = file.path(out_dir, paste0("V_raw_", tree_name, ".csv")),
        row.names = TRUE,
        col.names = TRUE,
        quote = FALSE
    )

    cat("Wrote:\n",
        " • metadata →", file.path(out_dir, paste0("metadata_", tree_name, ".csv")), "\n",
        " • raw phylo covariance →", file.path(out_dir, paste0("V_raw_", tree_name, ".csv")), "\n")

    # ─── 7. Speaker count transform (fixed: log + scale) ─────────────
    log_n_scaled <- scale(log(df$n_speakers))
    df$log_n_speakers_norm <- log_n_scaled[, 1]
    log_n_center <- as.numeric(attr(log_n_scaled, "scaled:center"))
    log_n_scale  <- as.numeric(attr(log_n_scaled, "scaled:scale"))

    base_df <- df
    coord_methods <- c("standard", "sqrt_plus_10", "log_plus_10")

    for (coord_method in coord_methods) {
        df_loop <- base_df

        lon_stats <- apply_coord_scaling(df_loop$longitude, coord_method)
        lat_stats <- apply_coord_scaling(df_loop$latitude, coord_method)

        df_loop$longitude_norm <- lon_stats$values
        df_loop$latitude_norm  <- lat_stats$values

        x_center <- lon_stats$center
        x_scale  <- lon_stats$scale
        y_center <- lat_stats$center
        y_scale  <- lat_stats$scale
        x_delta_original <- lon_stats$delta_original_per_sd
        y_delta_original <- lat_stats$delta_original_per_sd

        model_df <- df_loop
        model_df$log_rate_median <- log(model_df$rate_median)
        language_levels <- rownames(model_df)
        model_df$language_factor <- factor(language_levels, levels = language_levels)

        V <- V_raw[language_levels, language_levels]

        formula_str <- paste(
            "log_rate_median ~",
            "longitude_norm + latitude_norm + log_n_speakers_norm +",
            "longitude_norm:latitude_norm +",
            "latitude_norm:log_n_speakers_norm +",
            "longitude_norm:log_n_speakers_norm +",
            "(1|gr(language_factor, cov = V))"
        )
        formula_obj <- as.formula(formula_str)

        brm_args <- list(
            formula = formula_obj,
            data = model_df,
            family = gaussian(),
            refresh = 0,
            chains = 4,
            iter = 4000,
            warmup = 1000,
            seed = 20231103,
            cores = max(1L, min(4L, parallel::detectCores())),
            control = list(adapt_delta = 0.99, max_treedepth = 15),
            data2 = list(V = V)
        )

        cat(sprintf("\nFitting phylogenetic spatial regression (%s)...\n", coord_method))
        model <- do.call(brm, brm_args)
        model_type <- "brms_phylo"

        coef_df <- as.data.frame(brms::fixef(model, probs = c(0.025, 0.975)))
        colnames(coef_df) <- c("Estimate", "Est.Error", "CI_lower", "CI_upper")

        cat("\nPosterior summaries (normalized / transformed scale):\n")
        print(coef_df[, c("Estimate", "Est.Error", "CI_lower", "CI_upper")])

        # ─── R² and fitted values ─────────────────────────────────────
        r2_vals <- as.numeric(brms::bayes_R2(model))
        r_squared_summary <- list(
            mean  = mean(r2_vals),
            sd    = stats::sd(r2_vals),
            q2.5  = unname(stats::quantile(r2_vals, 0.025)),
            q97.5 = unname(stats::quantile(r2_vals, 0.975))
        )

        fitted_summary <- fitted(model, summary = TRUE, probs = c(0.025, 0.975))
        preds_df <- data.frame(
            language     = language_levels,
            fitted_mean  = fitted_summary[, "Estimate"],
            fitted_sd    = fitted_summary[, "Est.Error"],
            fitted_lower = fitted_summary[, "Q2.5"],
            fitted_upper = fitted_summary[, "Q97.5"],
            stringsAsFactors = FALSE
        )

        residuals <- model_df$log_rate_median - fitted_summary[, "Estimate"]
        rmse <- sqrt(mean(residuals^2, na.rm = TRUE))
        mae  <- mean(abs(residuals), na.rm = TRUE)

        loo_result <- tryCatch(
            brms::loo(model),
            error = function(e) {
                warning(sprintf("LOO computation failed for %s (%s): %s", tree_name, coord_method, e$message))
                NULL
            }
        )
        loo_metrics <- list(
            available   = !is.null(loo_result),
            elpd_loo    = NA_real_,
            elpd_loo_se = NA_real_,
            p_loo       = NA_real_,
            looic       = NA_real_,
            looic_se    = NA_real_
        )
        if (!is.null(loo_result)) {
            loo_metrics$elpd_loo    <- unname(loo_result$estimates["elpd_loo", "Estimate"])
            loo_metrics$elpd_loo_se <- unname(loo_result$estimates["elpd_loo", "SE"])
            loo_metrics$p_loo       <- unname(loo_result$estimates["p_loo", "Estimate"])
            loo_metrics$looic       <- unname(loo_result$estimates["looic", "Estimate"])
            loo_metrics$looic_se    <- unname(loo_result$estimates["looic", "SE"])
        }

        waic_result <- tryCatch(
            brms::waic(model),
            error = function(e) {
                warning(sprintf("WAIC computation failed for %s (%s): %s", tree_name, coord_method, e$message))
                NULL
            }
        )
        waic_metrics <- list(
            available = !is.null(waic_result),
            waic      = NA_real_,
            waic_se   = NA_real_,
            p_waic    = NA_real_
        )
        if (!is.null(waic_result)) {
            waic_metrics$waic    <- unname(waic_result$estimates["waic", "Estimate"])
            waic_metrics$waic_se <- unname(waic_result$estimates["waic", "SE"])
            waic_metrics$p_waic  <- unname(waic_result$estimates["p_waic", "Estimate"])
        }

        # ─── 5. Collect results into a list (JSON payload) ────────────
        results_list <- list(
            metadata             = as.data.frame(df_loop) %>% rownames_to_column("language"),
            coph                 = coph_df,
            model_type           = model_type,
            call                 = formula_str,
            coordinate_method    = coord_method,
            coordinate_transform = lon_stats$description,
            r_squared            = r_squared_summary,
            goodness_of_fit      = list(
                rmse = rmse,
                mae = mae,
                loo = loo_metrics,
                waic = waic_metrics
            ),
            summary              = coef_df %>% rownames_to_column("term"),
            preds                = preds_df,
            scaling_params = list(
                x_center                  = x_center,
                x_scale                   = x_scale,
                x_original_center         = lon_stats$original_center,
                x_delta_original_per_sd   = x_delta_original,
                y_center                  = y_center,
                y_scale                   = y_scale,
                y_original_center         = lat_stats$original_center,
                y_delta_original_per_sd   = y_delta_original,
                log_n_center              = log_n_center,
                log_n_scale               = log_n_scale,
                coordinate_method         = coord_method,
                coordinate_shift          = lon_stats$shift,
                coordinate_description    = lon_stats$description
            )
            # No "unstandardized_coefficients": we only report transformed/normalized effects
        )

        # ─── 6. Write JSON ────────────────────────────────────────────
        json_path <- file.path(
            out_dir,
            paste0("phylolm_results_", tree_name, "_", coord_method, ".json")
        )
        write_json(
            results_list,
            path       = json_path,
            pretty     = TRUE,
            auto_unbox = TRUE
        )

        cat("✔️ Results written to", json_path, "\n")

        # ─── 7. Save results as CSV (transformed scale only) ─────────
        csv_results <- data.frame(
            model             = model_type,
            tree              = tree_name,
            coordinate_method = coord_method,
            stringsAsFactors  = FALSE
        )

        # Add coefficient information (normalized / transformed scale)
        for (i in seq_len(nrow(coef_df))) {
            term_name       <- rownames(coef_df)[i]
            term_name_clean <- gsub("_norm", "", term_name)  # Remove _norm suffix
            term_name_clean <- gsub(":", "_", term_name_clean)  # Replace : with _

            csv_results[paste0("coef_", term_name_clean)]      <- coef_df[i, "Estimate"]
            csv_results[paste0("se_", term_name_clean)]        <- coef_df[i, "Est.Error"]
            csv_results[paste0("ci_lower_", term_name_clean)]  <- coef_df[i, "CI_lower"]
            csv_results[paste0("ci_upper_", term_name_clean)]  <- coef_df[i, "CI_upper"]
        }

        # Add all scaling parameters (for later interpretation / contrasts)
        csv_results$x_scale_sd               <- x_scale
        csv_results$y_scale_sd               <- y_scale
        csv_results$x_center                 <- x_center
        csv_results$y_center                 <- y_center
        csv_results$x_original_center        <- lon_stats$original_center
        csv_results$y_original_center        <- lat_stats$original_center
        csv_results$x_delta_original_per_sd  <- x_delta_original
        csv_results$y_delta_original_per_sd  <- y_delta_original
        csv_results$log_n_scale_sd           <- log_n_scale
        csv_results$log_n_center             <- log_n_center
        csv_results$coordinate_method_desc   <- lon_stats$description
        csv_results$coordinate_shift         <- lon_stats$shift
        csv_results$r2_mean                  <- r_squared_summary$mean
        csv_results$r2_sd                    <- r_squared_summary$sd
        csv_results$r2_q2_5                  <- r_squared_summary$q2.5
        csv_results$r2_q97_5                 <- r_squared_summary$q97.5
        csv_results$rmse                     <- rmse
        csv_results$mae                      <- mae
        csv_results$elpd_loo                 <- loo_metrics$elpd_loo
        csv_results$elpd_loo_se              <- loo_metrics$elpd_loo_se
        csv_results$p_loo                    <- loo_metrics$p_loo
        csv_results$looic                    <- loo_metrics$looic
        csv_results$looic_se                 <- loo_metrics$looic_se
        csv_results$waic                     <- waic_metrics$waic
        csv_results$waic_se                  <- waic_metrics$waic_se
        csv_results$p_waic                   <- waic_metrics$p_waic

        # Save CSV
        csv_path <- file.path(
            "speech_phylo/final_blr_results",
            paste0("phylolm_", tree_name, "_", coord_method, ".csv")
        )

        write.csv(csv_results, csv_path, row.names = FALSE)
        cat("✔️ CSV results written to", csv_path, "\n")
    }
}
