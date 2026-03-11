import re
from loguru import logger


def ndc_to_11(ndc: str) -> str | None:
    """Convert any NDC format to standard 11-digit 5-4-2."""
    if not ndc or not isinstance(ndc, str):
        return None

    clean = re.sub(r"[-\s]", "", ndc.strip())

    if len(clean) == 11:
        return clean

    if len(clean) == 10:
        parts = ndc.strip().split("-")
        if len(parts) == 3:
            labeler, product, package = parts
            return labeler.zfill(5) + product.zfill(4) + package.zfill(2)
        else:
            return "0" + clean

    if len(clean) < 10:
        return clean.zfill(11)

    return None


def ndc_from_components(labeler_code: str, product_code: str, package_size: str) -> str | None:
    """Build 11-digit NDC from SDUD component columns."""
    try:
        labeler = str(labeler_code).strip().zfill(5)
        product = str(product_code).strip().zfill(4)
        package = str(package_size).strip().zfill(2)
        return labeler + product + package
    except (ValueError, TypeError):
        return None


def harmonise_ndc_column(df, ndc_col="ndc"):
    """Standardise NDC column in a dataframe to 11-digit format."""
    if df[ndc_col].str.contains(";").any():
        df["ndc_primary"] = df[ndc_col].str.split(";").str[0].str.strip()
        df["ndc_11"] = df["ndc_primary"].apply(ndc_to_11)
    else:
        df["ndc_11"] = df[ndc_col].apply(ndc_to_11)

    matched = df["ndc_11"].notna().sum()
    total = len(df)
    rate = (matched / total) * 100 if total > 0 else 0
    logger.info(f"NDC harmonisation: {matched:,}/{total:,} matched ({rate:.1f}%)")

    return df