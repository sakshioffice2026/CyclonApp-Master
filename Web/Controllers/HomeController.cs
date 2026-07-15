using CyclonApp.Repositories.Contracts;
using CyclonApp.Repositories.Repositories;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Diagnostics;
using Microsoft.AspNetCore.Mvc;

namespace CyclonApp.Controllers;

public class HomeController : Controller
{
    public readonly IUnitOfWork _uow;

    public HomeController(IUnitOfWork uow)
    {
        _uow = uow;
    }

    [HttpGet]
    public IActionResult Index()
    {
        try
        {
            // Redirect to Projects dashboard if logged in
            if (User.Identity?.IsAuthenticated == true)
                return RedirectToAction("Index", "Dashboard");

            return RedirectToAction("Login", "Account");
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "HomeController",
                "Index",
                ex.ToString());

            return RedirectToAction("Login", "Account");
        }
    }

    [HttpGet]
    [AllowAnonymous]
    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        try
        {
            var feature = HttpContext.Features.Get<IExceptionHandlerPathFeature>();
            ViewBag.RequestId = System.Diagnostics.Activity.Current?.Id ?? HttpContext.TraceIdentifier;
            ViewBag.ErrorPath = feature?.Path;
            ViewBag.ErrorMessage = feature?.Error?.Message;
            return View();
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "HomeController",
                "Error",
                ex.ToString());

            return View();
        }
    }
}
